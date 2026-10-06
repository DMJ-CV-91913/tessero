/*
 * NumPy numerical routines on whole arrays (Tessero\Np): products (outer, inner, vdot, kron, cross, tensordot),
 * differences and integrals (diff, ediff1d, gradient, trapezoid, unwrap), interpolation (interp), sequences
 * (logspace, geomspace), Vandermonde matrices and the polynomial helpers (polyval, polyadd, polysub, polymul,
 * polyder, polyint, polydiv), windows (bartlett, blackman, hamming, hanning, kaiser), sinc and i0, pad,
 * nan_to_num and the complex-type predicates, and the in-place writers (put, place, putmask, copyto,
 * fill_diagonal, put_along_axis).
 *
 * Each follows NumPy 2.4's definition (numpy/_core/numeric.py, function_base.py, numpy/lib/_function_base_impl.py,
 * _polynomial_impl.py, _arraypad_impl.py, _type_check_impl.py, _index_tricks_impl.py, _shape_base_impl.py, and
 * the C routines behind interp and put): the same operation order, so float results round the same way; the same
 * result dtypes (NumPy 2 promotion, Python scalars weak where NumPy keeps them weak); the same errors. Products
 * that NumPy hands to BLAS (inner, vdot, tensordot on floats) are summed in index order here: the fixtures
 * compare them within the rounding bound of a length-n dot product (tools/parity/fixtures_np_numeric.py).
 */
#include "fn.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================ shared helpers (as in np_shape.c) */

typedef struct {
    tsr_array a;
    unsigned char s[16];
} harr;

static int is_intdt(int d) { return d == TSR_I64 || d == TSR_I32 || d == TSR_U8; }
static int is_intlike(int d) { return is_intdt(d) || d == TSR_BOOL; }
static int is_floatdt(int d) { return d == TSR_F64 || d == TSR_F32; }

static int promote2(int a, int b)
{
    if (a == b) return a;
    if (a == TSR_C128 || b == TSR_C128) return TSR_C128;
    if (a == TSR_BOOL) return b;
    if (b == TSR_BOOL) return a;
    if (a == TSR_F64 || b == TSR_F64) return TSR_F64;
    if (a == TSR_F32 || b == TSR_F32) { const int o = a == TSR_F32 ? b : a; return o == TSR_U8 ? TSR_F32 : TSR_F64; }
    if (a == TSR_I64 || b == TSR_I64) return TSR_I64;
    return TSR_I32;
}

static int promote_weak(int dt, char kind)
{
    if (kind == 'b') return dt;
    if (kind == 'i') return dt == TSR_BOOL ? TSR_I64 : dt;
    if (dt == TSR_BOOL || is_intdt(dt)) return TSR_F64;
    return dt;
}

static const char *dtname(int d)
{
    static const char *const N[] = {"float64", "float32", "int64", "int32", "uint8", "bool", "complex128"};
    return d >= 0 && d < TSR_NDTYPES ? N[d] : "?";
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
    h->a.ndim = 0;
    h->a.dtype = dt;
    return TSR_OK;
}

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

/* Python's int(x) of a number argument (truncation; NaN and infinity raise) */
static int int_trunc(const tsr_arg *a, const char *name, int64_t *out)
{
    if (a->kind == 1 && !(a->flags & 1)) {
        if (!(a->num > -9.2e18 && a->num < 9.2e18)) { fn_set_error(a->num != a->num ? "cannot convert float NaN to integer" : "cannot convert float infinity to integer"); return TSR_EARG; }
        *out = (int64_t)a->num;
        return TSR_OK;
    }
    if (a->kind == 3 && a->arr.ndim == 0 && is_floatdt(a->arr.dtype)) {
        const double x = rd_f64(elem0(&a->arr), a->arr.dtype);
        if (!(x > -9.2e18 && x < 9.2e18)) { fn_set_error("cannot convert float NaN to integer"); return TSR_EARG; }
        *out = (int64_t)x;
        return TSR_OK;
    }
    return int_param(a, name, out);
}

static int num_param(const tsr_arg *a, const char *name, double *out)
{
    if (a->kind == 1) { *out = (a->flags & 1) ? (double)a->ival : a->num; return TSR_OK; }
    if (a->kind == 4) { *out = a->num != 0; return TSR_OK; }
    if (a->kind == 3 && a->arr.ndim == 0 && a->arr.dtype != TSR_C128) { *out = rd_f64(elem0(&a->arr), a->arr.dtype); return TSR_OK; }
    fn_set_error("%s must be a real number", name);
    return TSR_EARG;
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

static int copy_view(const tsr_array *v, int dto, void *dst, const int64_t *dst_st)
{
    if (v->ndim == 0) {
        const int64_t one = 1, z = 0;
        return tsr_copy(v->dtype, dto, 1, &one, elem0(v), &z, dst, &z);
    }
    return tsr_copy(v->dtype, dto, v->ndim, v->shape, elem0(v), v->strides, dst, dst_st);
}

static int result_scalar(tsr_result *r, const char *p, int dt)
{
    switch (dt) {
    case TSR_F64: case TSR_F32: fn_result_num(r, rd_f64(p, dt)); return TSR_OK;
    case TSR_BOOL: memset(r, 0, sizeof *r); r->kind = 4; r->num = *(const uint8_t *)p != 0; return TSR_OK;
    case TSR_I64: case TSR_I32: case TSR_U8: fn_result_int(r, rd_i64(p, dt)); return TSR_OK;
    default: return TSR_EARG;
    }
}

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

static int bshape(int32_t na, const int64_t *sa, int32_t nb, const int64_t *sb, int32_t *nd, int64_t *out)
{
    int64_t a[TSR_MAXDIM], b[TSR_MAXDIM];
    memcpy(a, sa, sizeof(int64_t) * (size_t)na);
    memcpy(b, sb, sizeof(int64_t) * (size_t)nb);
    if (tsr_broadcast_shape(na, a, nb, b, nd, out) != 0) {
        fn_set_error("operands could not be broadcast together");
        return TSR_ESHAPE;
    }
    return TSR_OK;
}

static int v_broadcast(tsr_array *a, int32_t nd, const int64_t *sh)
{
    int64_t st[TSR_MAXDIM];
    if (tsr_broadcast_strides(a, nd, sh, st) != 0) {
        fn_set_error("operands could not be broadcast together");
        return TSR_ESHAPE;
    }
    a->ndim = nd;
    memcpy(a->shape, sh, sizeof(int64_t) * (size_t)nd);
    memcpy(a->strides, st, sizeof(int64_t) * (size_t)nd);
    return TSR_OK;
}

static void v_slice(tsr_array *a, int ax, int64_t start, int64_t stop)
{
    a->offset += start * a->strides[ax];
    a->shape[ax] = stop > start ? stop - start : 0;
}

/* ================================================================ owned arrays and element arithmetic */

typedef struct { tsr_array a; int64_t bytes; } own;          /* a C-contiguous tsr_alloc'ed array */

static int own_new(own *o, int dt, int32_t nd, const int64_t *sh)
{
    memset(o, 0, sizeof *o);
    const int64_t n = tsr_shape_size(nd, sh);
    if (n < 0) { fn_set_error("array is too big"); return TSR_EARG; }
    o->bytes = n * tsr_itemsize(dt) + 64;
    o->a.data = tsr_calloc(o->bytes);
    if (!o->a.data) return TSR_ENOMEM;
    tsr_array_init(&o->a, dt, nd, sh, o->a.data);
    return TSR_OK;
}

static void own_free(own *o) { if (o->a.data && o->bytes) tsr_free(o->a.data, o->bytes); o->a.data = NULL; o->bytes = 0; }

/* hand an owned array to a result (0-d -> a scalar) */
static int own_emit(tsr_result *r, own *o)
{
    if (o->a.ndim == 0 && o->a.dtype != TSR_C128) {
        int rc = result_scalar(r, elem0(&o->a), o->a.dtype);
        own_free(o);
        return rc;
    }
    memset(r, 0, sizeof *r);
    r->kind = 3;
    r->arr = o->a;
    r->bytes = o->bytes;
    o->a.data = NULL;
    o->bytes = 0;
    return TSR_OK;
}

/* a strided view as an owned contiguous copy of dtype dt */
static int own_from(own *o, const tsr_array *v, int dt)
{
    int rc = own_new(o, dt < 0 ? v->dtype : dt, v->ndim, v->shape);
    if (rc < 0) return rc;
    if (tsr_shape_size(v->ndim, v->shape) == 0) return TSR_OK;
    rc = copy_view(v, o->a.dtype, o->a.data, o->a.strides);
    if (rc < 0) own_free(o);
    return rc;
}

enum { OP_ADD, OP_SUB, OP_MUL, OP_TDIV, OP_NE };

/* the loop dtype of a binary operation (numpy's type resolution for add/subtract/multiply/true_divide) */
static int op_dtype(int op, int dt, int *out)
{
    if (op == OP_TDIV && (is_intlike(dt))) { *out = TSR_F64; return TSR_OK; }
    if (op == OP_SUB && dt == TSR_BOOL) {
        fn_set_error("numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.");
        return TSR_ETYPE;
    }
    if (op == OP_NE) { *out = TSR_BOOL; return TSR_OK; }
    *out = dt;
    return TSR_OK;
}

/* one element: z = x op y, x and y read as `in` dtype (already the loop dtype), written as `dt` */
static void elem_op(int op, int dt, const char *px, int dx, const char *py, int dy, char *pz)
{
    switch (dt) {
    case TSR_F64: {
        const double x = rd_f64(px, dx), y = rd_f64(py, dy);
        const double z = op == OP_ADD ? x + y : op == OP_SUB ? x - y : op == OP_MUL ? x * y : x / y;
        memcpy(pz, &z, 8);
        return;
    }
    case TSR_F32: {
        const float x = (float)rd_f64(px, dx), y = (float)rd_f64(py, dy);
        const float z = op == OP_ADD ? x + y : op == OP_SUB ? x - y : op == OP_MUL ? x * y : x / y;
        memcpy(pz, &z, 4);
        return;
    }
    case TSR_BOOL: {
        const int x = rd_i64(px, dx) != 0, y = rd_i64(py, dy) != 0;
        if (op == OP_NE) { *(uint8_t *)pz = (uint8_t)(rd_f64(px, dx) != rd_f64(py, dy)); return; }
        *(uint8_t *)pz = (uint8_t)(op == OP_MUL ? (x && y) : (x || y));
        return;
    }
    default: {
        const uint64_t x = (uint64_t)rd_i64(px, dx), y = (uint64_t)rd_i64(py, dy);
        const uint64_t z = op == OP_ADD ? x + y : op == OP_SUB ? x - y : x * y;
        if (dt == TSR_I64) { const int64_t v = (int64_t)z; memcpy(pz, &v, 8); }
        else if (dt == TSR_I32) { const int32_t v = (int32_t)(uint32_t)z; memcpy(pz, &v, 4); }
        else *(uint8_t *)pz = (uint8_t)z;
        return;
    }
    }
}

/* z = x op y with broadcasting; the loop dtype from the operands (wx / wy: weak Python scalar kinds or 0).
   `not_equal` compares in the promoted dtype. */
static int bop(int op, const tsr_array *x, char wx, const tsr_array *y, char wy, own *z)
{
    int dt;
    if (wx && !wy) dt = promote_weak(y->dtype, wx);
    else if (wy && !wx) dt = promote_weak(x->dtype, wy);
    else dt = promote2(x->dtype, y->dtype);
    if (dt == TSR_C128) { fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    int zt;
    int rc = op_dtype(op, dt, &zt);
    if (rc < 0) return rc;
    int32_t nd;
    int64_t sh[TSR_MAXDIM];
    if ((rc = bshape(x->ndim, x->shape, y->ndim, y->shape, &nd, sh)) < 0) return rc;
    tsr_array a = *x, b = *y;
    if ((rc = v_broadcast(&a, nd, sh)) < 0 || (rc = v_broadcast(&b, nd, sh)) < 0) return rc;
    if ((rc = own_new(z, zt, nd, sh)) < 0) return rc;
    const int64_t n = tsr_shape_size(nd, sh), zs = tsr_itemsize(zt);
    /* the operands are converted to the loop dtype first (numpy casts, then runs the typed loop) */
    const int ldt = op == OP_NE ? dt : zt;
    char *pz = (char *)z->a.data;
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t oa = 0, ob = 0;
        for (int d = 0; d < nd; d++) { oa += ix[d] * a.strides[d]; ob += ix[d] * b.strides[d]; }
        unsigned char ca[16], cb[16];
        const int64_t one = 1, zz = 0;
        tsr_copy(a.dtype, ldt, 1, &one, elem0(&a) + oa, &zz, ca, &zz);
        tsr_copy(b.dtype, ldt, 1, &one, elem0(&b) + ob, &zz, cb, &zz);
        if (op == OP_NE) {
            *(uint8_t *)(pz + q * zs) = (uint8_t)(is_floatdt(ldt) ? rd_f64((char *)ca, ldt) != rd_f64((char *)cb, ldt)
                                                                   : rd_i64((char *)ca, ldt) != rd_i64((char *)cb, ldt));
        } else elem_op(op, zt, (char *)ca, ldt, (char *)cb, ldt, pz + q * zs);
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < sh[d]) break; ix[d] = 0; }
    }
    return TSR_OK;
}

/* a double scalar as a 0-d array view (storage in h) */
static tsr_array scalar_f64(harr *h, double v)
{
    memset(h, 0, sizeof *h);
    memcpy(h->s, &v, 8);
    h->a.data = h->s;
    h->a.dtype = TSR_F64;
    return h->a;
}

static tsr_array scalar_i64(harr *h, int64_t v)
{
    memset(h, 0, sizeof *h);
    memcpy(h->s, &v, 8);
    h->a.data = h->s;
    h->a.dtype = TSR_I64;
    return h->a;
}

/* ravel a view into an owned contiguous copy (dtype kept) */
static int own_ravel(own *o, const tsr_array *v)
{
    int rc = own_from(o, v, -1);
    if (rc < 0) return rc;
    const int64_t n = tsr_shape_size(v->ndim, v->shape);
    o->a.ndim = 1;
    o->a.shape[0] = n;
    o->a.strides[0] = tsr_itemsize(o->a.dtype);
    return TSR_OK;
}

/* ================================================================ products */

/* outer(a, b): multiply(a.ravel()[:, None], b.ravel()[None, :]) */
static int r_outer(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr ha, hb;
    int rc = get_arr(argk(args, nargs, 0), &ha, "a");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &hb, "b");
    if (rc < 0) return rc;
    own ra, rb, z;
    if ((rc = own_ravel(&ra, &ha.a)) < 0) return rc;
    if ((rc = own_ravel(&rb, &hb.a)) < 0) { own_free(&ra); return rc; }
    tsr_array x = ra.a, y = rb.a;
    x.ndim = 2; x.shape[1] = 1; x.strides[1] = 0;
    y.ndim = 2; y.shape[1] = y.shape[0]; y.strides[1] = y.strides[0]; y.shape[0] = 1; y.strides[0] = 0;
    rc = bop(OP_MUL, &x, 0, &y, 0, &z);
    own_free(&ra);
    own_free(&rb);
    if (rc < 0) return rc;
    return own_emit(&res[0], &z);
}

/* sum_k x[k] * y[k] in dtype dt over n elements with strides (index order; ints wrap, bools or/and) */
static void dot_acc(int dt, int64_t n, const char *px, int64_t sx, int dx, const char *py, int64_t sy, int dy, char *out)
{
    if (dt == TSR_F64 || dt == TSR_F32) {
        double acc = 0.0;
        if (dt == TSR_F32) {
            float f = 0.0f;
            for (int64_t k = 0; k < n; k++) f += (float)rd_f64(px + k * sx, dx) * (float)rd_f64(py + k * sy, dy);
            memcpy(out, &f, 4);
            return;
        }
        for (int64_t k = 0; k < n; k++) acc += rd_f64(px + k * sx, dx) * rd_f64(py + k * sy, dy);
        memcpy(out, &acc, 8);
        return;
    }
    if (dt == TSR_BOOL) {
        int acc = 0;
        for (int64_t k = 0; k < n && !acc; k++) acc = rd_i64(px + k * sx, dx) && rd_i64(py + k * sy, dy);
        *(uint8_t *)out = (uint8_t)acc;
        return;
    }
    uint64_t acc = 0;
    for (int64_t k = 0; k < n; k++) acc += (uint64_t)rd_i64(px + k * sx, dx) * (uint64_t)rd_i64(py + k * sy, dy);
    if (dt == TSR_I64) { const int64_t v = (int64_t)acc; memcpy(out, &v, 8); }
    else if (dt == TSR_I32) { const int32_t v = (int32_t)(uint32_t)acc; memcpy(out, &v, 4); }
    else *(uint8_t *)out = (uint8_t)acc;
}

/* inner(a, b): sum over the last axes; a 0-d operand multiplies */
static int r_inner(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr ha, hb;
    int rc = get_arr(argk(args, nargs, 0), &ha, "a");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &hb, "b");
    if (rc < 0) return rc;
    const tsr_array *a = &ha.a, *b = &hb.a;
    if (a->ndim == 0 || b->ndim == 0) {
        own z;
        if ((rc = bop(OP_MUL, a, 0, b, 0, &z)) < 0) return rc;
        return own_emit(&res[0], &z);
    }
    const int64_t n = a->shape[a->ndim - 1];
    if (b->shape[b->ndim - 1] != n) { fn_set_error("shapes not aligned: dimension mismatch in the last axes"); return TSR_EARG; }
    const int dt = promote2(a->dtype, b->dtype);
    if (dt == TSR_C128) { fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    const int nd = a->ndim - 1 + b->ndim - 1;
    if (nd > TSR_MAXDIM) { fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    int64_t sh[TSR_MAXDIM];
    for (int d = 0; d < a->ndim - 1; d++) sh[d] = a->shape[d];
    for (int d = 0; d < b->ndim - 1; d++) sh[a->ndim - 1 + d] = b->shape[d];
    own z;
    if ((rc = own_new(&z, dt, nd, sh)) < 0) return rc;
    const int64_t na = tsr_shape_size(a->ndim - 1, a->shape), nb = tsr_shape_size(b->ndim - 1, b->shape), zs = tsr_itemsize(dt);
    int64_t ia[TSR_MAXDIM] = {0};
    char *pz = (char *)z.a.data;
    for (int64_t p = 0; p < na; p++) {
        int64_t oa = 0;
        for (int d = 0; d < a->ndim - 1; d++) oa += ia[d] * a->strides[d];
        int64_t ib[TSR_MAXDIM] = {0};
        for (int64_t q = 0; q < nb; q++) {
            int64_t ob = 0;
            for (int d = 0; d < b->ndim - 1; d++) ob += ib[d] * b->strides[d];
            dot_acc(dt, n, elem0(a) + oa, a->strides[a->ndim - 1], a->dtype, elem0(b) + ob, b->strides[b->ndim - 1], b->dtype, pz);
            pz += zs;
            for (int d = b->ndim - 2; d >= 0; d--) { if (++ib[d] < b->shape[d]) break; ib[d] = 0; }
        }
        for (int d = a->ndim - 2; d >= 0; d--) { if (++ia[d] < a->shape[d]) break; ia[d] = 0; }
    }
    return own_emit(&res[0], &z);
}

/* vdot(a, b): the flattened dot product */
static int r_vdot(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr ha, hb;
    int rc = get_arr(argk(args, nargs, 0), &ha, "a");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &hb, "b");
    if (rc < 0) return rc;
    own ra, rb;
    if ((rc = own_ravel(&ra, &ha.a)) < 0) return rc;
    if ((rc = own_ravel(&rb, &hb.a)) < 0) { own_free(&ra); return rc; }
    if (ra.a.shape[0] != rb.a.shape[0]) {
        own_free(&ra); own_free(&rb);
        fn_set_error("vectors have different lengths");
        return TSR_EARG;
    }
    const int dt = promote2(ra.a.dtype, rb.a.dtype);
    if (dt == TSR_C128) { own_free(&ra); own_free(&rb); fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    unsigned char out[16];
    dot_acc(dt, ra.a.shape[0], elem0(&ra.a), ra.a.strides[0], ra.a.dtype, elem0(&rb.a), rb.a.strides[0], rb.a.dtype, (char *)out);
    own_free(&ra);
    own_free(&rb);
    return result_scalar(&res[0], (const char *)out, dt);
}

/* kron(a, b) */
static int r_kron(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr ha, hb;
    int rc = get_arr(argk(args, nargs, 0), &ha, "a");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &hb, "b");
    if (rc < 0) return rc;
    tsr_array a = ha.a, b = hb.a;
    /* a = array(a, ndmin=b.ndim) */
    while (a.ndim < b.ndim) {
        memmove(a.shape + 1, a.shape, sizeof(int64_t) * (size_t)a.ndim);
        memmove(a.strides + 1, a.strides, sizeof(int64_t) * (size_t)a.ndim);
        a.shape[0] = 1;
        a.strides[0] = 0;
        a.ndim++;
    }
    const int nda = a.ndim, ndb = b.ndim, nd = nda > ndb ? nda : ndb;
    own z;
    if (nda == 0 || ndb == 0) {
        if ((rc = bop(OP_MUL, &a, 0, &b, 0, &z)) < 0) return rc;
        return own_emit(&res[0], &z);
    }
    if (2 * nd > TSR_MAXDIM) { fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    /* pad b to nd, then interleave: a -> (a0, 1, a1, 1, ...), b -> (1, b0, 1, b1, ...) */
    tsr_array bb = b;
    while (bb.ndim < nd) {
        memmove(bb.shape + 1, bb.shape, sizeof(int64_t) * (size_t)bb.ndim);
        memmove(bb.strides + 1, bb.strides, sizeof(int64_t) * (size_t)bb.ndim);
        bb.shape[0] = 1;
        bb.strides[0] = 0;
        bb.ndim++;
    }
    tsr_array ai = a, bi = bb;
    ai.ndim = bi.ndim = 2 * nd;
    for (int d = 0; d < nd; d++) {
        ai.shape[2 * d] = a.shape[d]; ai.strides[2 * d] = a.strides[d];
        ai.shape[2 * d + 1] = 1; ai.strides[2 * d + 1] = 0;
        bi.shape[2 * d] = 1; bi.strides[2 * d] = 0;
        bi.shape[2 * d + 1] = bb.shape[d]; bi.strides[2 * d + 1] = bb.strides[d];
    }
    if ((rc = bop(OP_MUL, &ai, 0, &bi, 0, &z)) < 0) return rc;
    /* reshape to as_ * bs */
    int64_t sh[TSR_MAXDIM] = {0};
    for (int d = 0; d < nd; d++) sh[d] = a.shape[d] * bb.shape[d];
    z.a.ndim = nd;
    for (int d = 0; d < nd; d++) z.a.shape[d] = sh[d];
    c_strides(nd, sh, tsr_itemsize(z.a.dtype), z.a.strides);
    return own_emit(&res[0], &z);
}

/* cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None) */
static int r_cross(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr ha, hb;
    int rc = get_arr(argk(args, nargs, 0), &ha, "a");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &hb, "b");
    if (rc < 0) return rc;
    int64_t xa = -1, xb = -1, xc = -1;
    if (!is_none(argk(args, nargs, 5))) {
        int64_t ax;
        if ((rc = int_param(argk(args, nargs, 5), "axis", &ax)) < 0) return rc;
        xa = xb = xc = ax;
    } else {
        if (!is_none(argk(args, nargs, 2)) && (rc = int_param(argk(args, nargs, 2), "axisa", &xa)) < 0) return rc;
        if (!is_none(argk(args, nargs, 3)) && (rc = int_param(argk(args, nargs, 3), "axisb", &xb)) < 0) return rc;
        if (!is_none(argk(args, nargs, 4)) && (rc = int_param(argk(args, nargs, 4), "axisc", &xc)) < 0) return rc;
    }
    tsr_array a = ha.a, b = hb.a;
    if (a.ndim < 1 || b.ndim < 1) { fn_set_error("At least one array has zero dimension"); return TSR_EARG; }
    int ia = 0, ib = 0;
    if (xa < -a.ndim || xa >= a.ndim) { fn_set_error("axisa: axis %lld is out of bounds for array of dimension %d", (long long)xa, a.ndim); return TSR_EARG; }
    if (xb < -b.ndim || xb >= b.ndim) { fn_set_error("axisb: axis %lld is out of bounds for array of dimension %d", (long long)xb, b.ndim); return TSR_EARG; }
    norm_axis(xa, a.ndim, &ia);
    norm_axis(xb, b.ndim, &ib);
    /* move the working axis to the end */
    {
        int64_t s = a.shape[ia], t = a.strides[ia];
        for (int d = ia; d < a.ndim - 1; d++) { a.shape[d] = a.shape[d + 1]; a.strides[d] = a.strides[d + 1]; }
        a.shape[a.ndim - 1] = s; a.strides[a.ndim - 1] = t;
        s = b.shape[ib]; t = b.strides[ib];
        for (int d = ib; d < b.ndim - 1; d++) { b.shape[d] = b.shape[d + 1]; b.strides[d] = b.strides[d + 1]; }
        b.shape[b.ndim - 1] = s; b.strides[b.ndim - 1] = t;
    }
    const int64_t na = a.shape[a.ndim - 1], nb = b.shape[b.ndim - 1];
    if ((na != 2 && na != 3) || (nb != 2 && nb != 3)) { fn_set_error("incompatible dimensions for cross product\n(dimension must be 2 or 3)"); return TSR_EARG; }
    const int dt = promote2(a.dtype, b.dtype);
    if (dt == TSR_C128) { fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    if (dt == TSR_BOOL) { int t; return op_dtype(OP_SUB, TSR_BOOL, &t); }
    /* the component arrays a0.. and b0..: views without the last axis */
    tsr_array ac[3], bc[3];
    for (int k = 0; k < na; k++) { ac[k] = a; ac[k].ndim--; ac[k].offset += k * a.strides[a.ndim - 1]; }
    for (int k = 0; k < nb; k++) { bc[k] = b; bc[k].ndim--; bc[k].offset += k * b.strides[b.ndim - 1]; }
    int32_t nd;
    int64_t sh[TSR_MAXDIM];
    if ((rc = bshape(ac[0].ndim, ac[0].shape, bc[0].ndim, bc[0].shape, &nd, sh)) < 0) return rc;
    const int three = na == 3 || nb == 3;
    int ic = 0;
    if (three) {
        if (nd + 1 > TSR_MAXDIM) { fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
        if (xc < -(nd + 1) || xc >= nd + 1) { fn_set_error("axisc: axis %lld is out of bounds for array of dimension %d", (long long)xc, nd + 1); return TSR_EARG; }
        norm_axis(xc, nd + 1, &ic);
    }
    for (int k = 0; k < na; k++) if ((rc = v_broadcast(&ac[k], nd, sh)) < 0) return rc;
    for (int k = 0; k < nb; k++) if ((rc = v_broadcast(&bc[k], nd, sh)) < 0) return rc;
    const int64_t n = tsr_shape_size(nd, sh);
    /* computed component-wise in dtype dt, in numpy's order: cp0 = a1*b2 - a2*b1, ... */
    own z;
    int64_t zsh[TSR_MAXDIM];
    memcpy(zsh, sh, sizeof(int64_t) * (size_t)nd);
    int znd = nd;
    if (three) zsh[znd++] = 3;
    if ((rc = own_new(&z, dt, znd, zsh)) < 0) return rc;
    const int64_t zs = tsr_itemsize(dt);
    char *pz = (char *)z.a.data;
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < n; q++) {
        char av[3][16], bv[3][16];
        const int64_t one = 1, zz = 0;
        for (int k = 0; k < na; k++) {
            int64_t o = 0;
            for (int d = 0; d < nd; d++) o += ix[d] * ac[k].strides[d];
            tsr_copy(a.dtype, dt, 1, &one, elem0(&ac[k]) + o, &zz, av[k], &zz);
        }
        for (int k = 0; k < nb; k++) {
            int64_t o = 0;
            for (int d = 0; d < nd; d++) o += ix[d] * bc[k].strides[d];
            tsr_copy(b.dtype, dt, 1, &one, elem0(&bc[k]) + o, &zz, bv[k], &zz);
        }
        char t1[16], t2[16], zero[16] = {0};
        char *out = pz + q * (three ? 3 : 1) * zs;
        if (na == 2 && nb == 2) {
            elem_op(OP_MUL, dt, av[0], dt, bv[1], dt, t1);
            elem_op(OP_MUL, dt, av[1], dt, bv[0], dt, t2);
            elem_op(OP_SUB, dt, t1, dt, t2, dt, out);
        } else if (na == 2) {                              /* b has 3: a2 = 0 */
            elem_op(OP_MUL, dt, av[1], dt, bv[2], dt, out);
            elem_op(OP_MUL, dt, av[0], dt, bv[2], dt, t1);
            elem_op(OP_SUB, dt, zero, dt, t1, dt, out + zs);         /* negative(cp1) */
            if (is_floatdt(dt)) { double v = -rd_f64(t1, dt); if (dt == TSR_F64) memcpy(out + zs, &v, 8); else { float f = (float)v; memcpy(out + zs, &f, 4); } }
            elem_op(OP_MUL, dt, av[0], dt, bv[1], dt, t1);
            elem_op(OP_MUL, dt, av[1], dt, bv[0], dt, t2);
            elem_op(OP_SUB, dt, t1, dt, t2, dt, out + 2 * zs);
        } else if (nb == 2) {                              /* a has 3: b2 = 0 */
            elem_op(OP_MUL, dt, av[2], dt, bv[1], dt, t1);
            elem_op(OP_SUB, dt, zero, dt, t1, dt, out);
            if (is_floatdt(dt)) { double v = -rd_f64(t1, dt); if (dt == TSR_F64) memcpy(out, &v, 8); else { float f = (float)v; memcpy(out, &f, 4); } }
            elem_op(OP_MUL, dt, av[2], dt, bv[0], dt, out + zs);
            elem_op(OP_MUL, dt, av[0], dt, bv[1], dt, t1);
            elem_op(OP_MUL, dt, av[1], dt, bv[0], dt, t2);
            elem_op(OP_SUB, dt, t1, dt, t2, dt, out + 2 * zs);
        } else {
            elem_op(OP_MUL, dt, av[1], dt, bv[2], dt, t1);
            elem_op(OP_MUL, dt, av[2], dt, bv[1], dt, t2);
            elem_op(OP_SUB, dt, t1, dt, t2, dt, out);
            elem_op(OP_MUL, dt, av[2], dt, bv[0], dt, t1);
            elem_op(OP_MUL, dt, av[0], dt, bv[2], dt, t2);
            elem_op(OP_SUB, dt, t1, dt, t2, dt, out + zs);
            elem_op(OP_MUL, dt, av[0], dt, bv[1], dt, t1);
            elem_op(OP_MUL, dt, av[1], dt, bv[0], dt, t2);
            elem_op(OP_SUB, dt, t1, dt, t2, dt, out + 2 * zs);
        }
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < sh[d]) break; ix[d] = 0; }
    }
    if (three && ic != znd - 1) {                        /* moveaxis(cp, -1, axisc) */
        tsr_array v = z.a;
        int perm[TSR_MAXDIM], k = 0;
        for (int d = 0; d < znd; d++) { if (d == ic) perm[d] = znd - 1; else perm[d] = k++; }
        int64_t s2[TSR_MAXDIM], t2[TSR_MAXDIM];
        for (int d = 0; d < znd; d++) { s2[d] = z.a.shape[perm[d]]; t2[d] = z.a.strides[perm[d]]; }
        memcpy(v.shape, s2, sizeof s2);
        memcpy(v.strides, t2, sizeof t2);
        rc = emit(&res[0], &v, -1);
        own_free(&z);
        return rc;
    }
    return own_emit(&res[0], &z);
}

/* tensordot(a, b, axes=2) */
static int r_tensordot(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr ha, hb;
    int rc = get_arr(argk(args, nargs, 0), &ha, "a");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &hb, "b");
    if (rc < 0) return rc;
    const tsr_arg *xa = argk(args, nargs, 2);
    int64_t axa[TSR_MAXDIM + 1], axb[TSR_MAXDIM + 1];
    int na = 0, nb = 0;
    const tsr_array *a = &ha.a, *b = &hb.a;
    if (is_none(xa) || xa->kind == 1 || xa->kind == 4 || (xa->kind == 3 && xa->arr.ndim == 0)) {
        int64_t k = 2;
        if (!is_none(xa) && (rc = int_param(xa, "axes", &k)) < 0) return rc;
        if (k < 0 || k > TSR_MAXDIM) {
            if (k < 0) { na = nb = 0; }
            else { fn_set_error("shape-mismatch for sum"); return TSR_EARG; }
        } else {
            na = nb = (int)k;
            for (int i = 0; i < k; i++) { axa[i] = -k + i; axb[i] = i; }
        }
    } else {
        /* a pair (axes_a, axes_b): each an int or a sequence */
        const tsr_arg *pa, *pb;
        tsr_arg ta, tb;
        if (xa->kind == 5 && xa->count == 2) { pa = &xa->items[0]; pb = &xa->items[1]; }
        else if (xa->kind == 3 && xa->arr.ndim >= 1 && xa->arr.shape[0] == 2
                   && (is_intlike(xa->arr.dtype) || tsr_shape_size(xa->arr.ndim, xa->arr.shape) == 0)) {
            /* [[], []] has no dtype of its own: an empty list becomes a float array */
            /* [[0, 1], [1, 0]] or [0, 1] arrives as an integer array */
            memset(&ta, 0, sizeof ta); memset(&tb, 0, sizeof tb);
            ta.kind = tb.kind = 3;
            ta.arr = tb.arr = xa->arr;
            ta.arr.ndim = tb.arr.ndim = xa->arr.ndim - 1;
            memmove(ta.arr.shape, xa->arr.shape + 1, sizeof(int64_t) * (size_t)(xa->arr.ndim - 1));
            memmove(ta.arr.strides, xa->arr.strides + 1, sizeof(int64_t) * (size_t)(xa->arr.ndim - 1));
            memcpy(tb.arr.shape, ta.arr.shape, sizeof ta.arr.shape);
            memcpy(tb.arr.strides, ta.arr.strides, sizeof ta.arr.strides);
            tb.arr.offset += xa->arr.strides[0];
            pa = &ta;
            pb = &tb;
        } else { fn_set_error("axes must be an integer or a pair of axis sequences"); return TSR_EARG; }
        const tsr_arg *pp[2] = {pa, pb};
        int64_t *outs[2] = {axa, axb};
        int *cnt[2] = {&na, &nb};
        for (int s = 0; s < 2; s++) {
            const tsr_arg *p = pp[s];
            if (p->kind == 5) {
                if (p->count > TSR_MAXDIM) { fn_set_error("too many axes"); return TSR_EARG; }
                for (int64_t i = 0; i < p->count; i++) if ((rc = int_param(&p->items[i], "axes", &outs[s][i])) < 0) return rc;
                *cnt[s] = (int)p->count;
            } else if (p->kind == 3 && p->arr.ndim == 1) {
                if (p->arr.shape[0] > TSR_MAXDIM) { fn_set_error("too many axes"); return TSR_EARG; }
                if (p->arr.shape[0] > 0 && !is_intlike(p->arr.dtype)) { fn_set_error("axes must be integers"); return TSR_EARG; }
                for (int64_t i = 0; i < p->arr.shape[0]; i++) outs[s][i] = rd_i64(elem0(&p->arr) + i * p->arr.strides[0], p->arr.dtype);
                *cnt[s] = (int)p->arr.shape[0];
            } else {
                if ((rc = int_param(p, "axes", &outs[s][0])) < 0) return rc;
                *cnt[s] = 1;
            }
        }
    }
    for (int i = 0; i < na; i++) for (int j = 0; j < i; j++) if (axa[i] == axa[j]) { fn_set_error("duplicate axes are not allowed in tensordot"); return TSR_EARG; }
    for (int i = 0; i < nb; i++) for (int j = 0; j < i; j++) if (axb[i] == axb[j]) { fn_set_error("duplicate axes are not allowed in tensordot"); return TSR_EARG; }
    int equal = na == nb;
    for (int k = 0; k < na && equal; k++) {
        if (axa[k] < -a->ndim || axa[k] >= a->ndim || axb[k] < -b->ndim || axb[k] >= b->ndim) {
            fn_set_error("tuple index out of range");
            return TSR_EINDEX;                            /* numpy: IndexError */
        }
        if (a->shape[axa[k] < 0 ? axa[k] + a->ndim : axa[k]] != b->shape[axb[k] < 0 ? axb[k] + b->ndim : axb[k]]) { equal = 0; break; }
        if (axa[k] < 0) axa[k] += a->ndim;
        if (axb[k] < 0) axb[k] += b->ndim;
    }
    if (!equal) { fn_set_error("shape-mismatch for sum"); return TSR_EARG; }
    const int dt = promote2(a->dtype, b->dtype);
    if (dt == TSR_C128) { fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    /* free axes of a, then of b */
    int fa[TSR_MAXDIM], fb[TSR_MAXDIM], nfa = 0, nfb = 0;
    for (int d = 0; d < a->ndim; d++) { int in = 0; for (int k = 0; k < na; k++) if (axa[k] == d) in = 1; if (!in) fa[nfa++] = d; }
    for (int d = 0; d < b->ndim; d++) { int in = 0; for (int k = 0; k < nb; k++) if (axb[k] == d) in = 1; if (!in) fb[nfb++] = d; }
    if (nfa + nfb > TSR_MAXDIM) { fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    int64_t sh[TSR_MAXDIM];
    for (int i = 0; i < nfa; i++) sh[i] = a->shape[fa[i]];
    for (int i = 0; i < nfb; i++) sh[nfa + i] = b->shape[fb[i]];
    own z;
    if ((rc = own_new(&z, dt, nfa + nfb, sh)) < 0) return rc;
    int64_t csh[TSR_MAXDIM];
    for (int k = 0; k < na; k++) csh[k] = a->shape[axa[k]];
    const int64_t nsum = tsr_shape_size(na, csh), nA = tsr_shape_size(nfa, sh), nB = tsr_shape_size(nfb, sh + nfa), zs = tsr_itemsize(dt);
    /* the summed axes flattened in C order of (axes_a...), as numpy's reshape of the transposed operands */
    const int64_t cb = (nsum + 1) * 16 + 64;
    int64_t *offs = (int64_t *)tsr_alloc(cb);
    if (!offs) { own_free(&z); return TSR_ENOMEM; }
    int64_t *offa = offs, *offb = offs + nsum + 1;
    {
        int64_t ix[TSR_MAXDIM] = {0};
        for (int64_t s = 0; s < nsum; s++) {
            int64_t oa = 0, ob = 0;
            for (int k = 0; k < na; k++) { oa += ix[k] * a->strides[axa[k]]; ob += ix[k] * b->strides[axb[k]]; }
            offa[s] = oa;
            offb[s] = ob;
            for (int k = na - 1; k >= 0; k--) { if (++ix[k] < csh[k]) break; ix[k] = 0; }
        }
    }
    char *pz = (char *)z.a.data;
    int64_t ia[TSR_MAXDIM] = {0};
    for (int64_t p = 0; p < nA; p++) {
        int64_t oa = 0;
        for (int i = 0; i < nfa; i++) oa += ia[i] * a->strides[fa[i]];
        int64_t ib[TSR_MAXDIM] = {0};
        for (int64_t q = 0; q < nB; q++) {
            int64_t ob = 0;
            for (int i = 0; i < nfb; i++) ob += ib[i] * b->strides[fb[i]];
            if (dt == TSR_F64) {
                double acc = 0.0;
                for (int64_t s = 0; s < nsum; s++) acc += rd_f64(elem0(a) + oa + offa[s], a->dtype) * rd_f64(elem0(b) + ob + offb[s], b->dtype);
                memcpy(pz, &acc, 8);
            } else if (dt == TSR_F32) {
                float acc = 0.0f;
                for (int64_t s = 0; s < nsum; s++) acc += (float)rd_f64(elem0(a) + oa + offa[s], a->dtype) * (float)rd_f64(elem0(b) + ob + offb[s], b->dtype);
                memcpy(pz, &acc, 4);
            } else if (dt == TSR_BOOL) {
                int acc = 0;
                for (int64_t s = 0; s < nsum && !acc; s++) acc = rd_i64(elem0(a) + oa + offa[s], a->dtype) && rd_i64(elem0(b) + ob + offb[s], b->dtype);
                *(uint8_t *)pz = (uint8_t)acc;
            } else {
                uint64_t acc = 0;
                for (int64_t s = 0; s < nsum; s++) acc += (uint64_t)rd_i64(elem0(a) + oa + offa[s], a->dtype) * (uint64_t)rd_i64(elem0(b) + ob + offb[s], b->dtype);
                if (dt == TSR_I64) { const int64_t v = (int64_t)acc; memcpy(pz, &v, 8); }
                else if (dt == TSR_I32) { const int32_t v = (int32_t)(uint32_t)acc; memcpy(pz, &v, 4); }
                else *(uint8_t *)pz = (uint8_t)acc;
            }
            pz += zs;
            for (int i = nfb - 1; i >= 0; i--) { if (++ib[i] < b->shape[fb[i]]) break; ib[i] = 0; }
        }
        for (int i = nfa - 1; i >= 0; i--) { if (++ia[i] < a->shape[fa[i]]) break; ia[i] = 0; }
    }
    tsr_free(offs, cb);
    return own_emit(&res[0], &z);
}

/* ================================================================ differences, integrals */

/* along axis ax of v: z = op(v[1:], v[:-1]) (numpy's diff step) */
static int diff_once(const tsr_array *v, int ax, own *z)
{
    tsr_array hi = *v, lo = *v;
    const int64_t n = v->shape[ax];
    v_slice(&hi, ax, n > 0 ? 1 : 0, n);
    v_slice(&lo, ax, 0, n > 0 ? n - 1 : 0);
    return bop(v->dtype == TSR_BOOL ? OP_NE : OP_SUB, &hi, 0, &lo, 0, z);
}

/* diff(a, n=1, axis=-1, prepend=None, append=None): None stands for numpy's "not given" */
static int r_diff(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    int64_t n = 1;
    if (!is_none(argk(args, nargs, 1)) && (rc = int_param(argk(args, nargs, 1), "n", &n)) < 0) return rc;
    if (n == 0) return emit(&res[0], &h.a, -1);
    if (n < 0) { fn_set_error("order must be non-negative but got %lld", (long long)n); return TSR_EARG; }
    if (h.a.ndim == 0) { fn_set_error("diff requires input that is at least one dimensional"); return TSR_EARG; }
    int64_t a64 = -1;
    int ax;
    if (!is_none(argk(args, nargs, 2)) && (rc = int_param(argk(args, nargs, 2), "axis", &a64)) < 0) return rc;
    if ((rc = norm_axis(a64, h.a.ndim, &ax)) < 0) return rc;
    own cur;
    memset(&cur, 0, sizeof cur);
    tsr_array v = h.a;
    const tsr_arg *pa = argk(args, nargs, 3), *aa = argk(args, nargs, 4);
    if (!is_none(pa) || !is_none(aa)) {
        /* concatenate((prepend, a, append), axis); a 0-d prepend/append is broadcast to a's shape with length 1 on axis */
        tsr_array parts[3];
        int np_ = 0;
        harr hp, hq;
        for (int s = 0; s < 2; s++) {
            const tsr_arg *x = s == 0 ? pa : aa;
            if (is_none(x)) { if (s == 0) parts[np_++] = v; continue; }
            harr *hh = s == 0 ? &hp : &hq;
            if ((rc = get_arr(x, hh, s == 0 ? "prepend" : "append")) < 0) return rc;
            tsr_array t = hh->a;
            if (t.ndim == 0) {
                int64_t sh[TSR_MAXDIM];
                memcpy(sh, v.shape, sizeof(int64_t) * (size_t)v.ndim);
                sh[ax] = 1;
                if ((rc = v_broadcast(&t, v.ndim, sh)) < 0) return rc;
            }
            if (s == 0) { parts[np_++] = t; parts[np_++] = v; }
            else parts[np_++] = t;
        }
        int dt = parts[0].dtype;
        for (int i = 0; i < np_; i++) {
            if (parts[i].ndim != v.ndim) { fn_set_error("all the input array dimensions except for the concatenation axis must match exactly"); return TSR_EARG; }
            for (int d = 0; d < v.ndim; d++) if (d != ax && parts[i].shape[d] != v.shape[d]) {
                fn_set_error("all the input array dimensions except for the concatenation axis must match exactly");
                return TSR_EARG;
            }
            dt = promote2(dt, parts[i].dtype);
        }
        int64_t sh[TSR_MAXDIM];
        memcpy(sh, v.shape, sizeof(int64_t) * (size_t)v.ndim);
        sh[ax] = 0;
        for (int i = 0; i < np_; i++) sh[ax] += parts[i].shape[ax];
        if ((rc = own_new(&cur, dt, v.ndim, sh)) < 0) return rc;
        int64_t at = 0;
        for (int i = 0; i < np_; i++) {
            if (tsr_shape_size(parts[i].ndim, parts[i].shape) > 0)
                if ((rc = copy_view(&parts[i], dt, (char *)cur.a.data + at * cur.a.strides[ax], cur.a.strides)) < 0) { own_free(&cur); return rc; }
            at += parts[i].shape[ax];
        }
        v = cur.a;
    }
    for (int64_t k = 0; k < n; k++) {
        own nx;
        if ((rc = diff_once(&v, ax, &nx)) < 0) { own_free(&cur); return rc; }
        own_free(&cur);
        cur = nx;
        v = cur.a;
        if (v.shape[ax] == 0) {                          /* further steps keep the empty axis */
            break;
        }
    }
    return own_emit(&res[0], &cur);
}

/* ediff1d(ary, to_end=None, to_begin=None) */
static int r_ediff1d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "ary");
    if (rc < 0) return rc;
    own r;
    if ((rc = own_ravel(&r, &h.a)) < 0) return rc;
    const int dt = r.a.dtype;
    const int64_t n = r.a.shape[0], ld = n > 1 ? n - 1 : 0;
    const tsr_arg *te = argk(args, nargs, 1), *tb = argk(args, nargs, 2);
    own parts[2];
    memset(parts, 0, sizeof parts);
    int64_t lens[2] = {0, 0};
    for (int s = 0; s < 2; s++) {
        const tsr_arg *x = s == 0 ? tb : te;
        if (is_none(x)) continue;
        harr hx;
        if ((rc = get_arr(x, &hx, s == 0 ? "to_begin" : "to_end")) < 0) goto fail;
        /* can_cast(to_x, ary.dtype, 'same_kind'); a Python scalar is weak (NEP 50 value-based check) */
        int src = hx.a.dtype;
        const char wk = weak_kind(x);
        int ok;
        if (wk == 'b') ok = 1;
        else if (wk == 'i') ok = dt != TSR_BOOL;
        else if (wk == 'f') ok = is_floatdt(dt) || dt == TSR_C128;
        else {
            static const char same_kind[7][8] = {"1100001", "1100001", "1111001", "1111001", "1111101", "1111111", "0000001"};
            ok = same_kind[src][dt] == '1';
        }
        if (!ok) {
            fn_set_error("dtype of `%s` must be compatible with input `ary` under the `same_kind` rule.", s == 0 ? "to_begin" : "to_end");
            rc = TSR_ETYPE;
            goto fail;
        }
        if ((rc = own_ravel(&parts[s], &hx.a)) < 0) goto fail;
        lens[s] = parts[s].a.shape[0];
    }
    {
        const int64_t total = lens[0] + ld + lens[1];
        own z;
        if ((rc = own_new(&z, dt, 1, &total)) < 0) goto fail;
        const int64_t isz = tsr_itemsize(dt);
        char *p = (char *)z.a.data;
        if (lens[0] > 0) copy_view(&parts[0].a, dt, p, z.a.strides);
        if (lens[1] > 0) copy_view(&parts[1].a, dt, p + (lens[0] + ld) * isz, z.a.strides);
        if (ld > 0) {
            own d;
            tsr_array hi = r.a, lo = r.a;
            v_slice(&hi, 0, 1, n);
            v_slice(&lo, 0, 0, n - 1);
            /* np.subtract(ary[1:], ary[:-1], out) in ary's dtype (bool raises) */
            if ((rc = bop(OP_SUB, &hi, 0, &lo, 0, &d)) < 0) { own_free(&z); goto fail; }
            copy_view(&d.a, dt, p + lens[0] * isz, z.a.strides);
            own_free(&d);
        }
        own_free(&r);
        own_free(&parts[0]);
        own_free(&parts[1]);
        return own_emit(&res[0], &z);
    }
fail:
    own_free(&r);
    own_free(&parts[0]);
    own_free(&parts[1]);
    return rc;
}

/* numpy.add.reduce of a contiguous float64 lane: pairwise when the reduced axis is the innermost one */
static double lane_sum(const double *x, int64_t n, int pairwise)
{
    if (n <= 0) return 0.0;
    if (pairwise) return tsr_psum(x, n);
    double acc = x[0];
    for (int64_t i = 1; i < n; i++) acc += x[i];
    return acc;
}

/* trapezoid(y, x=None, dx=1.0, axis=-1) */
static int r_trapezoid(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr hy;
    int rc = get_arr(argk(args, nargs, 0), &hy, "y");
    if (rc < 0) return rc;
    const tsr_array *y = &hy.a;
    int64_t a64 = -1;
    if (!is_none(argk(args, nargs, 3)) && (rc = int_param(argk(args, nargs, 3), "axis", &a64)) < 0) return rc;
    int ax;
    if (y->ndim == 0) { fn_set_error("tuple index out of range"); return TSR_EARG; }
    if ((rc = norm_axis(a64, y->ndim, &ax)) < 0) return rc;
    const int64_t n = y->shape[ax];
    /* s = y[1:] + y[:-1] */
    tsr_array y1 = *y, y0 = *y;
    v_slice(&y1, ax, n > 0 ? 1 : 0, n);
    v_slice(&y0, ax, 0, n > 0 ? n - 1 : 0);
    own s, t, u;
    if ((rc = bop(OP_ADD, &y1, 0, &y0, 0, &s)) < 0) return rc;
    const tsr_arg *xa = argk(args, nargs, 1);
    if (is_none(xa)) {
        const tsr_arg *da = argk(args, nargs, 2);
        harr hd;
        if (is_none(da)) { tsr_array d = scalar_f64(&hd, 1.0); rc = bop(OP_MUL, &d, 'f', &s.a, 0, &t); }
        else {
            if ((rc = get_arr(da, &hd, "dx")) < 0) { own_free(&s); return rc; }
            rc = bop(OP_MUL, &hd.a, weak_kind(da), &s.a, 0, &t);
        }
    } else {
        harr hx;
        if ((rc = get_arr(xa, &hx, "x")) < 0) { own_free(&s); return rc; }
        own d;
        tsr_array xv = hx.a;
        if (xv.ndim == 1) {
            if ((rc = diff_once(&xv, 0, &d)) < 0) { own_free(&s); return rc; }
            /* reshape to [1, ..., len, ..., 1] on axis */
            const int64_t len = d.a.shape[0], st = d.a.strides[0];
            d.a.ndim = y->ndim;
            for (int q = 0; q < y->ndim; q++) { d.a.shape[q] = 1; d.a.strides[q] = 0; }
            d.a.shape[ax] = len;
            d.a.strides[ax] = st;
        } else {
            int xax;
            if (xv.ndim == 0) { own_free(&s); fn_set_error("diff requires input that is at least one dimensional"); return TSR_EARG; }
            if ((rc = norm_axis(a64, xv.ndim, &xax)) < 0) { own_free(&s); return rc; }
            if ((rc = diff_once(&xv, xax, &d)) < 0) { own_free(&s); return rc; }
        }
        rc = bop(OP_MUL, &d.a, 0, &s.a, 0, &t);
        own_free(&d);
    }
    own_free(&s);
    if (rc < 0) return rc;
    harr h2;
    tsr_array two = scalar_f64(&h2, 2.0);
    rc = bop(OP_TDIV, &t.a, 0, &two, 'f', &u);
    own_free(&t);
    if (rc < 0) return rc;
    /* .sum(axis): u is C-contiguous; the reduction is pairwise when axis is its last axis */
    int rax = ax;
    if (u.a.ndim != y->ndim) { own_free(&u); fn_set_error("x and y shapes do not broadcast along the axis"); return TSR_EARG; }
    const int nd = u.a.ndim, dt = u.a.dtype;
    const int64_t len = u.a.shape[rax];
    int64_t osh[TSR_MAXDIM];
    int k = 0;
    for (int d = 0; d < nd; d++) if (d != rax) osh[k++] = u.a.shape[d];
    own z;
    const int rdt = is_intlike(dt) ? TSR_I64 : dt;
    if ((rc = own_new(&z, rdt, nd - 1, osh)) < 0) { own_free(&u); return rc; }
    const int64_t nout = tsr_shape_size(nd - 1, osh);
    const int64_t lb = (len + 1) * 8 + 64;
    double *lane = (double *)tsr_alloc(lb);
    if (!lane) { own_free(&u); own_free(&z); return TSR_ENOMEM; }
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t o = 0; o < nout; o++) {
        int64_t base = 0, kk = 0;
        for (int d = 0; d < nd; d++) if (d != rax) base += ix[kk++] * u.a.strides[d];
        for (int64_t i = 0; i < len; i++) lane[i] = rd_f64(elem0(&u.a) + base + i * u.a.strides[rax], dt);
        int inner = 1;                                   /* the reduced axis is the innermost non-trivial one */
        for (int d = rax + 1; d < nd; d++) if (u.a.shape[d] != 1) inner = 0;
        const double sum = lane_sum(lane, len, inner);
        if (rdt == TSR_F64) ((double *)z.a.data)[o] = sum;
        else if (rdt == TSR_F32) ((float *)z.a.data)[o] = (float)sum;
        else ((int64_t *)z.a.data)[o] = (int64_t)sum;
        for (int d = nd - 2; d >= 0; d--) { if (++ix[d] < osh[d]) break; ix[d] = 0; }
    }
    tsr_free(lane, lb);
    own_free(&u);
    return own_emit(&res[0], &z);
}

/* cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None): the running trapezoid integral
   (scipy.integrate.cumulative_trapezoid). The per-interval contributions are computed exactly as trapezoid,
   then accumulated along the axis instead of summed. Output axis length is n-1, or n when `initial` is given. */
static int r_cumulative_trapezoid(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    harr hy;
    int rc = get_arr(argk(args, nargs, 0), &hy, "y");
    if (rc < 0) return rc;
    const tsr_array *y = &hy.a;
    int64_t a64 = -1;
    if (!is_none(argk(args, nargs, 3)) && (rc = int_param(argk(args, nargs, 3), "axis", &a64)) < 0) return rc;
    int ax;
    if (y->ndim == 0) { fn_set_error("tuple index out of range"); return TSR_EARG; }
    if ((rc = norm_axis(a64, y->ndim, &ax)) < 0) return rc;
    const int64_t n = y->shape[ax];
    tsr_array y1 = *y, y0 = *y;
    v_slice(&y1, ax, n > 0 ? 1 : 0, n);
    v_slice(&y0, ax, 0, n > 0 ? n - 1 : 0);
    own s, t, u;
    if ((rc = bop(OP_ADD, &y1, 0, &y0, 0, &s)) < 0) return rc;
    const tsr_arg *xa = argk(args, nargs, 1);
    if (is_none(xa)) {
        const tsr_arg *da = argk(args, nargs, 2);
        harr hd;
        if (is_none(da)) { tsr_array d = scalar_f64(&hd, 1.0); rc = bop(OP_MUL, &d, 'f', &s.a, 0, &t); }
        else {
            if ((rc = get_arr(da, &hd, "dx")) < 0) { own_free(&s); return rc; }
            rc = bop(OP_MUL, &hd.a, weak_kind(da), &s.a, 0, &t);
        }
    } else {
        harr hx;
        if ((rc = get_arr(xa, &hx, "x")) < 0) { own_free(&s); return rc; }
        own d;
        tsr_array xv = hx.a;
        if (xv.ndim == 1) {
            if ((rc = diff_once(&xv, 0, &d)) < 0) { own_free(&s); return rc; }
            const int64_t len = d.a.shape[0], st = d.a.strides[0];
            d.a.ndim = y->ndim;
            for (int q = 0; q < y->ndim; q++) { d.a.shape[q] = 1; d.a.strides[q] = 0; }
            d.a.shape[ax] = len;
            d.a.strides[ax] = st;
        } else {
            int xax;
            if (xv.ndim == 0) { own_free(&s); fn_set_error("diff requires input that is at least one dimensional"); return TSR_EARG; }
            if ((rc = norm_axis(a64, xv.ndim, &xax)) < 0) { own_free(&s); return rc; }
            if ((rc = diff_once(&xv, xax, &d)) < 0) { own_free(&s); return rc; }
        }
        rc = bop(OP_MUL, &d.a, 0, &s.a, 0, &t);
        own_free(&d);
    }
    own_free(&s);
    if (rc < 0) return rc;
    harr h2;
    tsr_array two = scalar_f64(&h2, 2.0);
    rc = bop(OP_TDIV, &t.a, 0, &two, 'f', &u);
    own_free(&t);
    if (rc < 0) return rc;
    if (u.a.ndim != y->ndim) { own_free(&u); fn_set_error("x and y shapes do not broadcast along the axis"); return TSR_EARG; }
    const int nd = u.a.ndim, dt = u.a.dtype;
    const int64_t m = u.a.shape[ax];
    const tsr_arg *ia = argk(args, nargs, 4);
    const int has_init = !is_none(ia);
    double init = 0.0;
    if (has_init && (rc = num_param(ia, "initial", &init)) < 0) { own_free(&u); return rc; }
    if (has_init && init != 0.0) { own_free(&u); fn_set_error("`initial` must be `None` or `0`."); return TSR_EARG; }
    int64_t osh[TSR_MAXDIM];
    for (int d = 0; d < nd; d++) osh[d] = u.a.shape[d];
    osh[ax] = has_init ? m + 1 : m;
    own z;
    if ((rc = own_new(&z, TSR_F64, nd, osh)) < 0) { own_free(&u); return rc; }
    double *out = (double *)z.a.data;
    const int64_t zax = z.a.strides[ax] / (int64_t)sizeof(double);
    int64_t nlane = 1;
    for (int d = 0; d < nd; d++) if (d != ax) nlane *= u.a.shape[d];
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t o = 0; o < nlane; o++) {
        int64_t ubase = 0, zbase = 0;
        for (int d = 0; d < nd; d++) if (d != ax) { ubase += ix[d] * u.a.strides[d]; zbase += ix[d] * (z.a.strides[d] / (int64_t)sizeof(double)); }
        double running = 0.0;
        if (has_init) out[zbase] = init;
        for (int64_t j = 0; j < m; j++) {
            running += rd_f64(elem0(&u.a) + ubase + j * u.a.strides[ax], dt);
            out[zbase + (has_init ? j + 1 : j) * zax] = running;
        }
        for (int d = nd - 1; d >= 0; d--) { if (d == ax) continue; if (++ix[d] < u.a.shape[d]) break; ix[d] = 0; }
    }
    own_free(&u);
    return own_emit(&res[0], &z);
}

/* composite Simpson over the triples (start, start+2, ...) up to stop, for one lane (scipy _basic_simpson);
   uniform spacing dx when x is NULL, else the irregular-spacing form with per-triple spacings from x. */
static double basic_simpson_lane(const double *y, const double *x, int64_t start, int64_t stop, double dx)
{
    double acc = 0.0;
    if (!x) {
        for (int64_t i = start; i < stop; i += 2) acc += y[i] + 4.0 * y[i + 1] + y[i + 2];
        acc *= dx / 3.0;
    } else {
        for (int64_t i = start; i < stop; i += 2) {
            const double h0 = x[i + 1] - x[i], h1 = x[i + 2] - x[i + 1];
            const double hsum = h0 + h1, hprod = h0 * h1;
            const double h0divh1 = h1 != 0.0 ? h0 / h1 : 0.0;
            const double inv = h0divh1 != 0.0 ? 1.0 / h0divh1 : 0.0;
            const double mid = hprod != 0.0 ? hsum * (hsum / hprod) : 0.0;
            acc += hsum / 6.0 * (y[i] * (2.0 - inv) + y[i + 1] * mid + y[i + 2] * (2.0 - h0divh1));
        }
    }
    return acc;
}

/* the composite Simpson's rule over one lane of length N (scipy.integrate.simpson). For an even number of
   samples (odd number of intervals) the last interval uses Cartwright's correction, matching scipy. */
static double simpson_lane(const double *y, const double *x, int64_t N, double dx)
{
    if (N < 2) return 0.0;
    if (N % 2 == 0) {
        if (N == 2) { const double h = x ? (x[1] - x[0]) : dx; return 0.5 * h * (y[1] + y[0]); }
        double result = basic_simpson_lane(y, x, 0, N - 3, dx);
        const double h0 = x ? (x[N - 2] - x[N - 3]) : dx;
        const double h1 = x ? (x[N - 1] - x[N - 2]) : dx;
        double num, den;
        num = 2.0 * h1 * h1 + 3.0 * h0 * h1; den = 6.0 * (h1 + h0);
        const double alpha = den != 0.0 ? num / den : 0.0;
        num = h1 * h1 + 3.0 * h0 * h1; den = 6.0 * h0;
        const double beta = den != 0.0 ? num / den : 0.0;
        num = h1 * h1 * h1; den = 6.0 * h0 * (h0 + h1);
        const double eta = den != 0.0 ? num / den : 0.0;
        result += alpha * y[N - 1] + beta * y[N - 2] - eta * y[N - 3];
        return result;
    }
    return basic_simpson_lane(y, x, 0, N - 2, dx);
}

/* simpson(y, x=None, dx=1.0, axis=-1): the composite Simpson's rule, reduced over the axis
   (scipy.integrate.simpson). */
static int r_simpson(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    harr hy;
    int rc = get_arr(argk(args, nargs, 0), &hy, "y");
    if (rc < 0) return rc;
    if (hy.a.ndim == 0) { fn_set_error("tuple index out of range"); return TSR_EARG; }
    own ysrc;
    if ((rc = own_from(&ysrc, &hy.a, TSR_F64)) < 0) return rc;
    const tsr_array *y = &ysrc.a;
    const int nd = y->ndim;
    int64_t a64 = -1;
    if (!is_none(argk(args, nargs, 3)) && (rc = int_param(argk(args, nargs, 3), "axis", &a64)) < 0) { own_free(&ysrc); return rc; }
    int ax;
    if ((rc = norm_axis(a64, nd, &ax)) < 0) { own_free(&ysrc); return rc; }
    const int64_t N = y->shape[ax];
    double dx = 1.0;
    own xsrc;
    int have_x = 0, x_is_1d = 0;
    const tsr_array *x = NULL;
    const tsr_arg *xa = argk(args, nargs, 1);
    if (!is_none(xa)) {
        harr hx;
        if ((rc = get_arr(xa, &hx, "x")) < 0) { own_free(&ysrc); return rc; }
        if ((rc = own_from(&xsrc, &hx.a, TSR_F64)) < 0) { own_free(&ysrc); return rc; }
        x = &xsrc.a; have_x = 1;
        if (x->ndim == 1) {
            x_is_1d = 1;
            if (x->shape[0] != N) { own_free(&ysrc); own_free(&xsrc); fn_set_error("If given, length of x along axis must be the same as y."); return TSR_EARG; }
        } else if (x->ndim != nd) {
            own_free(&ysrc); own_free(&xsrc); fn_set_error("If given, shape of x must be 1-D or the same as y."); return TSR_EARG;
        } else if (x->shape[ax] != N) {
            own_free(&ysrc); own_free(&xsrc); fn_set_error("If given, length of x along axis must be the same as y."); return TSR_EARG;
        }
    } else {
        const tsr_arg *da = argk(args, nargs, 2);
        if (!is_none(da) && (rc = num_param(da, "dx", &dx)) < 0) { own_free(&ysrc); return rc; }
    }
    int64_t osh[TSR_MAXDIM];
    int k = 0;
    for (int d = 0; d < nd; d++) if (d != ax) osh[k++] = y->shape[d];
    own z;
    if ((rc = own_new(&z, TSR_F64, nd - 1, osh)) < 0) { own_free(&ysrc); if (have_x) own_free(&xsrc); return rc; }
    double *out = (double *)z.a.data;
    const int64_t nout = tsr_shape_size(nd - 1, osh);
    const int64_t bufb = (N > 0 ? N : 1) * (int64_t)sizeof(double) + 64;
    double *yl = (double *)tsr_alloc(bufb);
    double *xl = have_x ? (double *)tsr_alloc(bufb) : NULL;
    if (!yl || (have_x && !xl)) { if (yl) tsr_free(yl, bufb); if (xl) tsr_free(xl, bufb); own_free(&ysrc); if (have_x) own_free(&xsrc); own_free(&z); return TSR_ENOMEM; }
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t o = 0; o < nout; o++) {
        int64_t ybase = 0, xbase = 0, kk = 0;
        for (int d = 0; d < nd; d++) if (d != ax) { ybase += ix[kk] * y->strides[d]; if (have_x && !x_is_1d) xbase += ix[kk] * x->strides[d]; kk++; }
        for (int64_t i = 0; i < N; i++) yl[i] = rd_f64(elem0(y) + ybase + i * y->strides[ax], TSR_F64);
        if (have_x) {
            if (x_is_1d) for (int64_t i = 0; i < N; i++) xl[i] = rd_f64(elem0(x) + i * x->strides[0], TSR_F64);
            else for (int64_t i = 0; i < N; i++) xl[i] = rd_f64(elem0(x) + xbase + i * x->strides[ax], TSR_F64);
        }
        out[o] = simpson_lane(yl, have_x ? xl : NULL, N, dx);
        for (int d = nd - 2; d >= 0; d--) { if (++ix[d] < osh[d]) break; ix[d] = 0; }
    }
    tsr_free(yl, bufb);
    if (xl) tsr_free(xl, bufb);
    own_free(&ysrc);
    if (have_x) own_free(&xsrc);
    return own_emit(&res[0], &z);
}

/* numpy's float remainder (npy_divmod's mod: the sign of the divisor) */
static double py_fmod(double a, double b)
{
    double mod = fmod(a, b);
    if (b == 0) return mod;
    if (mod != 0) { if ((b < 0) != (mod < 0)) mod += b; }
    else mod = copysign(0.0, b);
    return mod;
}

/* unwrap(p, discont=None, axis=-1, period=2*pi) */
static int r_unwrap(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr hp;
    int rc = get_arr(argk(args, nargs, 0), &hp, "p");
    if (rc < 0) return rc;
    const tsr_array *p = &hp.a;
    if (p->ndim == 0) { fn_set_error("diff requires input that is at least one dimensional"); return TSR_EARG; }
    int64_t a64 = -1;
    int ax;
    if (!is_none(argk(args, nargs, 2)) && (rc = int_param(argk(args, nargs, 2), "axis", &a64)) < 0) return rc;
    if ((rc = norm_axis(a64, p->ndim, &ax)) < 0) return rc;
    const tsr_arg *pa = argk(args, nargs, 3);
    double period = 2 * M_PI;
    char pk = 'f';
    int64_t iperiod = 0;
    if (!is_none(pa)) {
        if ((rc = num_param(pa, "period", &period)) < 0) return rc;
        pk = weak_kind(pa);
        if (pa->kind == 1 && (pa->flags & 1)) iperiod = pa->ival;
        if (pa->kind == 3) pk = 0;
    }
    own dd;
    if ((rc = diff_once(p, ax, &dd)) < 0) return rc;
    /* dtype = result_type(dd, period) */
    int dt = dd.a.dtype;
    if (pk) dt = promote_weak(dt, pk);
    else dt = promote2(dt, pa->arr.dtype);
    if (dt == TSR_C128) { own_free(&dd); fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    const int integer = is_intdt(dt);
    const tsr_arg *da = argk(args, nargs, 1);
    double discont;
    if (is_none(da)) discont = integer ? (double)iperiod / 2 : period / 2;
    else if ((rc = num_param(da, "discont", &discont)) < 0) { own_free(&dd); return rc; }
    /* up = array(p, dtype=dtype); up[slice1] = p[slice1] + cumsum(ph_correct) */
    own up;
    if ((rc = own_from(&up, p, dt)) < 0) { own_free(&dd); return rc; }
    const int nd = p->ndim;
    const int64_t n = p->shape[ax];
    int64_t osh[TSR_MAXDIM];
    int k = 0;
    for (int d = 0; d < nd; d++) if (d != ax) osh[k++] = p->shape[d];
    const int64_t nlanes = tsr_shape_size(nd - 1, osh);
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t o = 0; o < nlanes && n > 1; o++) {
        int64_t bd = 0, bu = 0, bp = 0, kk = 0;
        for (int d = 0; d < nd; d++) if (d != ax) { bd += ix[kk] * dd.a.strides[d]; bu += ix[kk] * up.a.strides[d]; bp += ix[kk] * p->strides[d]; kk++; }
        if (!integer) {
            const double ihigh = period / 2, ilow = -ihigh;
            double csum = 0.0;
            for (int64_t i = 0; i < n - 1; i++) {
                const double d = rd_f64(elem0(&dd.a) + bd + i * dd.a.strides[ax], dd.a.dtype);
                double ddmod = py_fmod(d - ilow, period) + ilow;
                if (ddmod == ilow && d > 0) ddmod = ihigh;       /* boundary_ambiguous: always for floats */
                double ph = ddmod - d;
                if (fabs(d) < discont) ph = 0.0;
                csum += ph;
                const double pv = rd_f64(elem0(p) + bp + (i + 1) * p->strides[ax], p->dtype);
                const double v = pv + csum;
                char *q = (char *)up.a.data + bu + (i + 1) * up.a.strides[ax];
                if (dt == TSR_F64) memcpy(q, &v, 8); else { const float f = (float)v; memcpy(q, &f, 4); }
            }
        } else {
            /* integer arithmetic: divmod(period, 2); np.mod floor semantics; cumsum in int64 */
            const int64_t P = iperiod;
            if (P == 0) { own_free(&dd); own_free(&up); fn_set_error("integer modulo by zero"); return TSR_EARG; }
            int64_t ihigh = P / 2, rem = P % 2;
            if (rem != 0 && ((rem < 0) != (2 < 0))) { rem += 2; ihigh -= 1; }
            const int amb = rem == 0;
            const int64_t ilow = -ihigh;
            uint64_t csum = 0;
            for (int64_t i = 0; i < n - 1; i++) {
                const int64_t d = rd_i64(elem0(&dd.a) + bd + i * dd.a.strides[ax], dd.a.dtype);
                int64_t m = (int64_t)((uint64_t)d - (uint64_t)ilow) % P;
                if (m != 0 && ((m < 0) != (P < 0))) m += P;
                int64_t ddmod = m + ilow;
                if (amb && ddmod == ilow && d > 0) ddmod = ihigh;
                int64_t ph = (int64_t)((uint64_t)ddmod - (uint64_t)d);
                if (fabs((double)d) < discont) ph = 0;
                csum += (uint64_t)ph;
                const int64_t pv = rd_i64(elem0(p) + bp + (i + 1) * p->strides[ax], p->dtype);
                const int64_t v = (int64_t)((uint64_t)pv + csum);
                char *q = (char *)up.a.data + bu + (i + 1) * up.a.strides[ax];
                if (dt == TSR_I64) memcpy(q, &v, 8);
                else if (dt == TSR_I32) { const int32_t w = (int32_t)v; memcpy(q, &w, 4); }
                else *(uint8_t *)q = (uint8_t)v;
            }
        }
        for (int d = nd - 2; d >= 0; d--) { if (++ix[d] < osh[d]) break; ix[d] = 0; }
    }
    own_free(&dd);
    return own_emit(&res[0], &up);
}

/* ================================================================ interp */

#define LIKELY_IN_CACHE_SIZE 8

/* numpy/_core/src/multiarray/compiled_base.c: binary_search_with_guess */
static int64_t bsearch_guess(double key, const double *arr, int64_t len, int64_t guess)
{
    int64_t imin = 0, imax = len;
    if (key > arr[len - 1]) return len;
    else if (key < arr[0]) return -1;
    if (len <= 4) {
        int64_t i;
        for (i = 1; i < len && key >= arr[i]; ++i);
        return i - 1;
    }
    if (guess > len - 3) guess = len - 3;
    if (guess < 1) guess = 1;
    if (key < arr[guess]) {
        if (key < arr[guess - 1]) {
            imax = guess - 1;
            if (guess > LIKELY_IN_CACHE_SIZE && key >= arr[guess - LIKELY_IN_CACHE_SIZE]) imin = guess - LIKELY_IN_CACHE_SIZE;
        } else return guess - 1;
    } else {
        if (key < arr[guess + 1]) return guess;
        else {
            if (key < arr[guess + 2]) return guess + 1;
            else {
                imin = guess + 2;
                if (guess < len - LIKELY_IN_CACHE_SIZE - 1 && key < arr[guess + LIKELY_IN_CACHE_SIZE]) imax = guess + LIKELY_IN_CACHE_SIZE;
            }
        }
    }
    while (imin < imax) {
        const int64_t imid = imin + ((imax - imin) >> 1);
        if (key >= arr[imid]) imin = imid + 1;
        else imax = imid;
    }
    return imin - 1;
}

/* a float64 copy of an array argument (any shape), flat C order */
static int f64_flat(const tsr_array *v, double **out, int64_t *n, int64_t *bytes)
{
    *n = tsr_shape_size(v->ndim, v->shape);
    if (*n < 0) return TSR_ENOMEM;
    if (v->dtype == TSR_C128) { fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    *bytes = (*n + 1) * 8 + 64;
    *out = (double *)tsr_alloc(*bytes);
    if (!*out) return TSR_ENOMEM;
    if (*n == 0) return TSR_OK;
    int64_t st[TSR_MAXDIM];
    c_strides(v->ndim, v->shape, 8, st);
    return copy_view(v, TSR_F64, *out, st);
}

static void argsort_f64_stable(const double *v, int64_t n, int64_t *ord, int64_t *tmp)
{
    for (int64_t i = 0; i < n; i++) ord[i] = i;
    for (int64_t w = 1; w < n; w *= 2) {
        for (int64_t lo = 0; lo < n; lo += 2 * w) {
            int64_t mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, a = lo, b = mid, k = lo;
            while (a < mid && b < hi) {
                const double x = v[ord[a]], y = v[ord[b]];
                const int bl = (y < x) || (x != x && y == y);        /* NaN last */
                tmp[k++] = bl ? ord[b++] : ord[a++];
            }
            while (a < mid) tmp[k++] = ord[a++];
            while (b < hi) tmp[k++] = ord[b++];
        }
        memcpy(ord, tmp, sizeof(int64_t) * (size_t)n);
    }
}

/* interp(x, xp, fp, left=None, right=None, period=None) */
static int r_interp(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr hx, hxp, hfp;
    int rc = get_arr(argk(args, nargs, 0), &hx, "x");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &hxp, "xp");
    if (rc == 0) rc = get_arr(argk(args, nargs, 2), &hfp, "fp");
    if (rc < 0) return rc;
    if (hfp.a.dtype == TSR_C128) { fn_set_error("complex fp is not supported"); return TSR_ETYPE; }
    double *x = NULL, *xp = NULL, *fp = NULL;
    int64_t nx, nxp, nfp, bx = 0, bxp = 0, bfp = 0;
    if ((rc = f64_flat(&hx.a, &x, &nx, &bx)) < 0) return rc;
    if ((rc = f64_flat(&hxp.a, &xp, &nxp, &bxp)) < 0) { tsr_free(x, bx); return rc; }
    if ((rc = f64_flat(&hfp.a, &fp, &nfp, &bfp)) < 0) { tsr_free(x, bx); tsr_free(xp, bxp); return rc; }
    const tsr_arg *pa = argk(args, nargs, 5);
    double lval = 0, rval = 0;
    int have_l = 0, have_r = 0;
    own z;
    memset(&z, 0, sizeof z);
    if (!is_none(pa)) {
        double period;
        if ((rc = num_param(pa, "period", &period)) < 0) goto out;
        if (period == 0) { fn_set_error("period must be a non-zero value"); rc = TSR_EARG; goto out; }
        period = fabs(period);
        if (hxp.a.ndim != 1 || hfp.a.ndim != 1) { fn_set_error("Data points must be 1-D sequences"); rc = TSR_EARG; goto out; }
        if (nxp != nfp) { fn_set_error("fp and xp are not of the same length"); rc = TSR_EARG; goto out; }
        for (int64_t i = 0; i < nx; i++) x[i] = py_fmod(x[i], period);
        for (int64_t i = 0; i < nxp; i++) xp[i] = py_fmod(xp[i], period);
        /* sort xp (with fp), then extend by one period on both sides */
        const int64_t ob = (nxp + 1) * 16 + 64, nb2 = (nxp + 2) * 16 + 64;
        int64_t *ord = (int64_t *)tsr_alloc(ob);
        double *ext = (double *)tsr_alloc(nb2);
        if (!ord || !ext) { if (ord) tsr_free(ord, ob); if (ext) tsr_free(ext, nb2); rc = TSR_ENOMEM; goto out; }
        argsort_f64_stable(xp, nxp, ord, ord + nxp + 1);
        double *nxv = ext, *nfv = ext + nxp + 2;
        for (int64_t i = 0; i < nxp; i++) { nxv[i + 1] = xp[ord[i]]; nfv[i + 1] = fp[ord[i]]; }
        if (nxp > 0) {
            nxv[0] = nxv[nxp] - period; nfv[0] = nfv[nxp];
            nxv[nxp + 1] = nxv[1] + period; nfv[nxp + 1] = nfv[1];
        }
        tsr_free(ord, ob);
        tsr_free(xp, bxp);
        tsr_free(fp, bfp);
        xp = ext;
        fp = ext + nxp + 2;
        bxp = nb2;
        bfp = 0;
        nxp = nfp = nxp > 0 ? nxp + 2 : 0;
    } else {
        const tsr_arg *la = argk(args, nargs, 3), *ra = argk(args, nargs, 4);
        if (!is_none(la)) { if ((rc = num_param(la, "left", &lval)) < 0) goto out; have_l = 1; }
        if (!is_none(ra)) { if ((rc = num_param(ra, "right", &rval)) < 0) goto out; have_r = 1; }
        if (hxp.a.ndim != 1) { fn_set_error(hxp.a.ndim == 0 ? "object of too small depth for desired array" : "object too deep for desired array"); rc = TSR_EARG; goto out; }
        if (hfp.a.ndim != 1) { fn_set_error(hfp.a.ndim == 0 ? "object of too small depth for desired array" : "object too deep for desired array"); rc = TSR_EARG; goto out; }
    }
    if (nxp == 0) { fn_set_error("array of sample points is empty"); rc = TSR_EARG; goto out; }
    if (nfp != nxp) { fn_set_error("fp and xp are not of the same length."); rc = TSR_EARG; goto out; }
    if (!have_l) lval = fp[0];
    if (!have_r) rval = fp[nxp - 1];
    if ((rc = own_new(&z, TSR_F64, hx.a.ndim, hx.a.shape)) < 0) goto out;
    {
        double *dres = (double *)z.a.data;
        if (nxp == 1) {
            for (int64_t i = 0; i < nx; i++) dres[i] = x[i] < xp[0] ? lval : (x[i] > xp[0] ? rval : fp[0]);
        } else {
            int64_t j = 0;
            for (int64_t i = 0; i < nx; i++) {
                const double xv = x[i];
                if (xv != xv) { dres[i] = xv; continue; }
                j = bsearch_guess(xv, xp, nxp, j);
                if (j == -1) dres[i] = lval;
                else if (j == nxp) dres[i] = rval;
                else if (j == nxp - 1) dres[i] = fp[j];
                else if (xp[j] == xv) dres[i] = fp[j];
                else {
                    const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
                    dres[i] = slope * (xv - xp[j]) + fp[j];
                    if (dres[i] != dres[i]) {
                        dres[i] = slope * (xv - xp[j + 1]) + fp[j + 1];
                        if (dres[i] != dres[i] && fp[j] == fp[j + 1]) dres[i] = fp[j];
                    }
                }
            }
        }
    }
out:
    tsr_free(x, bx);
    if (xp) tsr_free(xp, bxp);
    if (fp && bfp) tsr_free(fp, bfp);
    if (rc < 0) { own_free(&z); return rc; }
    return own_emit(&res[0], &z);
}

/* ================================================================ gradient */

/* gradient(f, *varargs, axis=None, edge_order=1) */
static int r_gradient(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr hf;
    int rc = get_arr(argk(args, nargs, 0), &hf, "f");
    if (rc < 0) return rc;
    const tsr_array *f = &hf.a;
    if (f->dtype == TSR_C128) { fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    const int N = f->ndim;
    int axes[TSR_MAXDIM + 1], nax = 0;
    const tsr_arg *aa = argk(args, nargs, 2);
    if (is_none(aa)) { for (int d = 0; d < N; d++) axes[nax++] = d; }
    else {
        int64_t v[TSR_MAXDIM + 1];
        int n;
        if (aa->kind == 5) {
            if (aa->count > TSR_MAXDIM) { fn_set_error("too many axes"); return TSR_EARG; }
            for (int64_t i = 0; i < aa->count; i++) if ((rc = int_param(&aa->items[i], "axis", &v[i])) < 0) return rc;
            n = (int)aa->count;
        } else if (aa->kind == 3 && aa->arr.ndim == 1) {
            n = (int)aa->arr.shape[0];
            if (n > TSR_MAXDIM) { fn_set_error("too many axes"); return TSR_EARG; }
            if (n > 0 && !is_intlike(aa->arr.dtype)) { fn_set_error("axis must be integers"); return TSR_EARG; }
            for (int i = 0; i < n; i++) v[i] = rd_i64(elem0(&aa->arr) + i * aa->arr.strides[0], aa->arr.dtype);
        } else {
            if ((rc = int_param(aa, "axis", &v[0])) < 0) return rc;
            n = 1;
        }
        for (int i = 0; i < n; i++) {
            if ((rc = norm_axis(v[i], N, &axes[i])) < 0) return rc;
            for (int j = 0; j < i; j++) if (axes[j] == axes[i]) { fn_set_error("repeated axis"); return TSR_EARG; }
        }
        nax = n;
    }
    int64_t eo = 1;
    if (!is_none(argk(args, nargs, 3)) && (rc = int_param(argk(args, nargs, 3), "edge_order", &eo)) < 0) return rc;
    /* spacings: none -> 1.0; one scalar -> for every axis; else one scalar or 1-D coordinate array per axis */
    const tsr_arg *va = argk(args, nargs, 1);
    const int64_t nv = va->kind == 5 ? va->count : 0;
    double sdx[TSR_MAXDIM];
    double *adx[TSR_MAXDIM];                              /* non-uniform spacings (diff of coordinates) or NULL */
    int64_t adxb[TSR_MAXDIM];
    memset(adx, 0, sizeof adx);
    for (int i = 0; i < nax; i++) sdx[i] = 1.0;
    if (nv == 1) {
        harr h;
        if ((rc = get_arr(&va->items[0], &h, "varargs")) < 0) return rc;
        if (h.a.ndim == 0) { const double d = rd_f64(elem0(&h.a), h.a.dtype); for (int i = 0; i < nax; i++) sdx[i] = d; }
        else if (nax != 1) { fn_set_error("invalid number of arguments"); return TSR_ETYPE; }
    }
    if (nv > 1 && nv != nax) { fn_set_error("invalid number of arguments"); return TSR_ETYPE; }
    if (nv >= 1 && !(nv == 1 && nax != 1)) {
        for (int i = 0; i < nv && i < nax; i++) {
            harr h;
            if ((rc = get_arr(&va->items[i], &h, "varargs")) < 0) goto fail;
            if (h.a.ndim == 0) { sdx[i] = rd_f64(elem0(&h.a), h.a.dtype); continue; }
            if (h.a.ndim != 1) { fn_set_error("distances must be either scalars or 1d"); rc = TSR_EARG; goto fail; }
            if (h.a.shape[0] != f->shape[axes[i]]) { fn_set_error("when 1d, distances must match the length of the corresponding dimension"); rc = TSR_EARG; goto fail; }
            if (h.a.dtype == TSR_C128) { fn_set_error("complex arrays are not supported here"); rc = TSR_ETYPE; goto fail; }
            const int64_t m = h.a.shape[0];
            /* diffx = np.diff(distances) (integers as float64; bools: not_equal) */
            if (h.a.dtype == TSR_BOOL) { fn_set_error("boolean coordinates are not supported"); rc = TSR_ETYPE; goto fail; }
            adxb[i] = (m + 1) * 8 + 64;
            adx[i] = (double *)tsr_alloc(adxb[i]);
            if (!adx[i]) { rc = TSR_ENOMEM; goto fail; }
            for (int64_t k = 0; k + 1 < m; k++) {
                const double hi = rd_f64(elem0(&h.a) + (k + 1) * h.a.strides[0], h.a.dtype), lo = rd_f64(elem0(&h.a) + k * h.a.strides[0], h.a.dtype);
                adx[i][k] = h.a.dtype == TSR_F32 ? (double)((float)hi - (float)lo) : hi - lo;
            }
            /* constant spacing reduces to the scalar case: (diffx == diffx[0]).all() */
            int same = m >= 2;
            for (int64_t k = 1; k + 1 < m && same; k++) if (!(adx[i][k] == adx[i][0])) same = 0;
            if (m < 2) { fn_set_error("index 0 is out of bounds for axis 0 with size 0"); rc = TSR_EARG; goto fail; }
            if (same) { sdx[i] = adx[i][0]; tsr_free(adx[i], adxb[i]); adx[i] = NULL; }
        }
    }
    if (eo > 2) { fn_set_error("'edge_order' greater than 2 not supported"); rc = TSR_EARG; goto fail; }
    const int otype = f->dtype == TSR_F32 ? TSR_F32 : TSR_F64;
    if (!tsr_fits((double)nax * (double)tsr_shape_size(N, f->shape) * (double)tsr_itemsize(otype))) {
        fn_set_error("Unable to allocate the %d gradient arrays", nax);
        rc = TSR_ENOMEM;
        goto fail;
    }
    own outs[TSR_MAXDIM];
    memset(outs, 0, sizeof outs);
    for (int i = 0; i < nax; i++) {
        const int ax = axes[i];
        const int64_t n = f->shape[ax];
        if (n < eo + 1) { fn_set_error("Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required."); rc = TSR_EARG; break; }
        if (f->dtype == TSR_BOOL) { int t; rc = op_dtype(OP_SUB, TSR_BOOL, &t); break; }
        if ((rc = own_new(&outs[i], otype, N, f->shape)) < 0) break;
        /* each lane along ax */
        int64_t osh[TSR_MAXDIM];
        int k = 0;
        for (int d = 0; d < N; d++) if (d != ax) osh[k++] = f->shape[d];
        const int64_t nl = tsr_shape_size(N - 1, osh);
        int64_t ix[TSR_MAXDIM] = {0};
        const double *ad = adx[i];
        const double dx = sdx[i];
        for (int64_t l = 0; l < nl; l++) {
            int64_t bf = 0, bo = 0, kk = 0;
            for (int d = 0; d < N; d++) if (d != ax) { bf += ix[kk] * f->strides[d]; bo += ix[kk] * outs[i].a.strides[d]; kk++; }
#define FV(j) rd_f64(elem0(f) + bf + (j) * f->strides[ax], f->dtype)
#define OUT(j, v) do { const double v_ = (v); char *q_ = (char *)outs[i].a.data + bo + (j) * outs[i].a.strides[ax]; \
                       if (otype == TSR_F64) memcpy(q_, &v_, 8); else { const float w_ = (float)v_; memcpy(q_, &w_, 4); } } while (0)
            for (int64_t j = 1; j + 1 < n; j++) {
                if (!ad) OUT(j, (FV(j + 1) - FV(j - 1)) / (2. * dx));
                else {
                    const double dx1 = ad[j - 1], dx2 = ad[j];
                    const double a = -(dx2) / (dx1 * (dx1 + dx2));
                    const double b = (dx2 - dx1) / (dx1 * dx2);
                    const double c = dx1 / (dx2 * (dx1 + dx2));
                    OUT(j, a * FV(j - 1) + b * FV(j) + c * FV(j + 1));
                }
            }
            if (eo == 1) {
                const double d0 = ad ? ad[0] : dx, dn = ad ? ad[n - 2] : dx;
                OUT(0, (FV(1) - FV(0)) / d0);
                OUT(n - 1, (FV(n - 1) - FV(n - 2)) / dn);
            } else {
                double a, b, c;
                if (!ad) { a = -1.5 / dx; b = 2. / dx; c = -0.5 / dx; }
                else {
                    const double dx1 = ad[0], dx2 = ad[1];
                    a = -(2. * dx1 + dx2) / (dx1 * (dx1 + dx2));
                    b = (dx1 + dx2) / (dx1 * dx2);
                    c = -dx1 / (dx2 * (dx1 + dx2));
                }
                OUT(0, a * FV(0) + b * FV(1) + c * FV(2));
                if (!ad) { a = 0.5 / dx; b = -2. / dx; c = 1.5 / dx; }
                else {
                    const double dx1 = ad[n - 3], dx2 = ad[n - 2];
                    a = (dx2) / (dx1 * (dx1 + dx2));
                    b = -(dx2 + dx1) / (dx1 * dx2);
                    c = (2. * dx2 + dx1) / (dx2 * (dx1 + dx2));
                }
                OUT(n - 1, a * FV(n - 3) + b * FV(n - 2) + c * FV(n - 1));
            }
#undef FV
#undef OUT
            for (int d = N - 2; d >= 0; d--) { if (++ix[d] < osh[d]) break; ix[d] = 0; }
        }
    }
    if (rc == 0) {
        if (nax == 1) rc = own_emit(&res[0], &outs[0]);
        else {
            tsr_result *items = fn_result_seq(&res[0], nax);
            if (!items) rc = TSR_ENOMEM;
            for (int i = 0; i < nax && rc == 0; i++) rc = own_emit(&items[i], &outs[i]);
        }
    }
    for (int i = 0; i < nax; i++) own_free(&outs[i]);
fail:
    for (int i = 0; i < TSR_MAXDIM; i++) if (adx[i]) tsr_free(adx[i], adxb[i]);
    return rc;
}

/* ================================================================ sequences */

/* numpy.linspace over broadcast float64 start/stop: result shape (num, *bshape) before the axis move */
static int linspace_core(const tsr_array *start, const tsr_array *stop, int64_t num, int endpoint, own *y)
{
    int32_t nd;
    int64_t sh[TSR_MAXDIM];
    int rc = bshape(start->ndim, start->shape, stop->ndim, stop->shape, &nd, sh);
    if (rc < 0) return rc;
    if (nd + 1 > TSR_MAXDIM) { fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    tsr_array a = *start, b = *stop;
    if ((rc = v_broadcast(&a, nd, sh)) < 0 || (rc = v_broadcast(&b, nd, sh)) < 0) return rc;
    int64_t ysh[TSR_MAXDIM];
    ysh[0] = num;
    memcpy(ysh + 1, sh, sizeof(int64_t) * (size_t)nd);
    if ((rc = own_new(y, TSR_F64, nd + 1, ysh)) < 0) return rc;
    const int64_t div = endpoint ? num - 1 : num, m = tsr_shape_size(nd, sh);
    const int64_t db = (m + 1) * 16 + 64;
    double *delta = (double *)tsr_alloc(db);
    if (!delta) { own_free(y); return TSR_ENOMEM; }
    double *st = delta + m + 1;
    int any_zero = 0;
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < m; q++) {
        int64_t oa = 0, ob = 0;
        for (int d = 0; d < nd; d++) { oa += ix[d] * a.strides[d]; ob += ix[d] * b.strides[d]; }
        const double s0 = rd_f64(elem0(&a) + oa, a.dtype), s1 = rd_f64(elem0(&b) + ob, b.dtype);
        delta[q] = s1 - s0;
        st[q] = s0;
        if (div > 0 && delta[q] / (double)div == 0) any_zero = 1;
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < sh[d]) break; ix[d] = 0; }
    }
    double *p = (double *)y->a.data;
    for (int64_t i = 0; i < num; i++)
        for (int64_t q = 0; q < m; q++) {
            double v = (double)i;
            if (div > 0) {
                if (any_zero) { v = v / (double)div; v = v * delta[q]; }
                else v = v * (delta[q] / (double)div);
            } else v = v * delta[q];
            p[i * m + q] = v + st[q];
        }
    if (endpoint && num > 1) {
        int64_t jx[TSR_MAXDIM] = {0};
        for (int64_t q = 0; q < m; q++) {
            int64_t ob = 0;
            for (int d = 0; d < nd; d++) ob += jx[d] * b.strides[d];
            p[(num - 1) * m + q] = rd_f64(elem0(&b) + ob, b.dtype);
            for (int d = nd - 1; d >= 0; d--) { if (++jx[d] < sh[d]) break; jx[d] = 0; }
        }
    }
    tsr_free(delta, db);
    return TSR_OK;
}

/* move axis 0 of an owned array to `axis` (a copy), then cast to dt */
static int finish_seq(tsr_result *r, own *y, int64_t axis_raw, int dt)
{
    const int nd = y->a.ndim;
    int ax;
    int rc = norm_axis(axis_raw, nd, &ax);
    if (rc < 0) { own_free(y); return rc; }
    tsr_array v = y->a;
    if (ax != 0) {
        const int64_t s0 = v.shape[0], t0 = v.strides[0];
        for (int d = 0; d < ax; d++) { v.shape[d] = v.shape[d + 1]; v.strides[d] = v.strides[d + 1]; }
        v.shape[ax] = s0;
        v.strides[ax] = t0;
    }
    if (dt < 0) dt = TSR_F64;
    if (is_intdt(dt)) {                                  /* integer dtype: floor first (numpy.linspace) */
        double *p = (double *)y->a.data;
        const int64_t n = tsr_shape_size(nd, y->a.shape);
        for (int64_t i = 0; i < n; i++) p[i] = floor(p[i]);
    }
    rc = emit(r, &v, dt);
    own_free(y);
    return rc;
}

static int dtype_arg(const tsr_arg *a, int *dt)
{
    *dt = -1;
    if (is_none(a)) return TSR_OK;
    static const struct { const char *n; int d; } T[] = {
        {"float64", TSR_F64}, {"float", TSR_F64}, {"f8", TSR_F64}, {"float32", TSR_F32}, {"f4", TSR_F32},
        {"int64", TSR_I64}, {"int", TSR_I64}, {"i8", TSR_I64}, {"int32", TSR_I32}, {"i4", TSR_I32},
        {"uint8", TSR_U8}, {"u1", TSR_U8}, {"bool", TSR_BOOL}, {"complex128", TSR_C128}, {"complex", TSR_C128},
    };
    if (a->kind == 2 && a->str)
        for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) if (strcmp(a->str, T[i].n) == 0) { *dt = T[i].d; return TSR_OK; }
    fn_set_error("data type '%s' not understood", a->kind == 2 && a->str ? a->str : "?");
    return TSR_EARG;
}

/* logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0) */
static int r_logspace(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h0, h1, hb;
    int rc = get_arr(argk(args, nargs, 0), &h0, "start");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &h1, "stop");
    if (rc < 0) return rc;
    int64_t num = 50, axis = 0;
    if (!is_none(argk(args, nargs, 2)) && (rc = int_param(argk(args, nargs, 2), "num", &num)) < 0) return rc;
    if (num < 0) { fn_set_error("Number of samples, %lld, must be non-negative.", (long long)num); return TSR_EARG; }
    const tsr_arg *ea = argk(args, nargs, 3);
    const int endpoint = is_none(ea) || ea->num != 0;
    const tsr_arg *ba = argk(args, nargs, 4);
    if (is_none(ba)) { tsr_array b = scalar_f64(&hb, 10.0); hb.a = b; }
    else if ((rc = get_arr(ba, &hb, "base")) < 0) return rc;
    int dt;
    if ((rc = dtype_arg(argk(args, nargs, 5), &dt)) < 0) return rc;
    if (!is_none(argk(args, nargs, 6)) && (rc = int_param(argk(args, nargs, 6), "axis", &axis)) < 0) return rc;
    if (h0.a.dtype == TSR_C128 || h1.a.dtype == TSR_C128 || hb.a.dtype == TSR_C128) { fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    tsr_array s0 = h0.a, s1 = h1.a, bb = hb.a;
    if (bb.ndim > 0) {
        /* start, stop, base broadcast with ndmin = their broadcast ndim; base gets an axis at `axis` */
        int32_t nd = 0;
        int64_t sh[TSR_MAXDIM];
        if ((rc = bshape(s0.ndim, s0.shape, s1.ndim, s1.shape, &nd, sh)) < 0 || (rc = bshape(nd, sh, bb.ndim, bb.shape, &nd, sh)) < 0) return rc;
        tsr_array *ops[3] = {&s0, &s1, &bb};
        for (int k = 0; k < 3; k++)
            while (ops[k]->ndim < nd) {
                memmove(ops[k]->shape + 1, ops[k]->shape, sizeof(int64_t) * (size_t)ops[k]->ndim);
                memmove(ops[k]->strides + 1, ops[k]->strides, sizeof(int64_t) * (size_t)ops[k]->ndim);
                ops[k]->shape[0] = 1; ops[k]->strides[0] = 0; ops[k]->ndim++;
            }
        int bax;
        if ((rc = norm_axis(axis, bb.ndim + 1, &bax)) < 0) return rc;
        memmove(bb.shape + bax + 1, bb.shape + bax, sizeof(int64_t) * (size_t)(bb.ndim - bax));
        memmove(bb.strides + bax + 1, bb.strides + bax, sizeof(int64_t) * (size_t)(bb.ndim - bax));
        bb.shape[bax] = 1; bb.strides[bax] = 0; bb.ndim++;
    }
    own y;
    if ((rc = linspace_core(&s0, &s1, num, endpoint, &y)) < 0) return rc;
    /* move axis 0 to `axis` for y, then power(base, y) */
    int ax;
    if ((rc = norm_axis(axis, y.a.ndim, &ax)) < 0) { own_free(&y); return rc; }
    tsr_array yv = y.a;
    if (ax != 0) {
        const int64_t s = yv.shape[0], t = yv.strides[0];
        for (int d = 0; d < ax; d++) { yv.shape[d] = yv.shape[d + 1]; yv.strides[d] = yv.strides[d + 1]; }
        yv.shape[ax] = s; yv.strides[ax] = t;
    }
    int32_t nd;
    int64_t sh[TSR_MAXDIM];
    if ((rc = bshape(bb.ndim, bb.shape, yv.ndim, yv.shape, &nd, sh)) < 0) { own_free(&y); return rc; }
    tsr_array B = bb, Y = yv;
    if ((rc = v_broadcast(&B, nd, sh)) < 0 || (rc = v_broadcast(&Y, nd, sh)) < 0) { own_free(&y); return rc; }
    own z;
    const int pdt = promote2(bb.dtype, TSR_F64) == TSR_F32 ? TSR_F32 : TSR_F64;
    if ((rc = own_new(&z, pdt, nd, sh)) < 0) { own_free(&y); return rc; }
    const int64_t n = tsr_shape_size(nd, sh);
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t ob = 0, oy = 0;
        for (int d = 0; d < nd; d++) { ob += ix[d] * B.strides[d]; oy += ix[d] * Y.strides[d]; }
        const double v = pow(rd_f64(elem0(&B) + ob, B.dtype), rd_f64(elem0(&Y) + oy, TSR_F64));
        if (pdt == TSR_F64) ((double *)z.a.data)[q] = v; else ((float *)z.a.data)[q] = (float)v;
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < sh[d]) break; ix[d] = 0; }
    }
    own_free(&y);
    if (dt >= 0 && dt != z.a.dtype) {
        rc = emit(&res[0], &z.a, dt);
        own_free(&z);
        return rc;
    }
    return own_emit(&res[0], &z);
}

static double sign_d(double x) { return x > 0 ? 1.0 : x < 0 ? -1.0 : x == 0 ? 0.0 : x; }

/* geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0) */
static int r_geomspace(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h0, h1;
    int rc = get_arr(argk(args, nargs, 0), &h0, "start");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &h1, "stop");
    if (rc < 0) return rc;
    if (h0.a.dtype == TSR_C128 || h1.a.dtype == TSR_C128) { fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    int64_t num = 50, axis = 0;
    if (!is_none(argk(args, nargs, 2)) && (rc = int_param(argk(args, nargs, 2), "num", &num)) < 0) return rc;
    const tsr_arg *ea = argk(args, nargs, 3);
    const int endpoint = is_none(ea) || ea->num != 0;
    int dt;
    if ((rc = dtype_arg(argk(args, nargs, 4), &dt)) < 0) return rc;
    if (!is_none(argk(args, nargs, 5)) && (rc = int_param(argk(args, nargs, 5), "axis", &axis)) < 0) return rc;
    own s0, s1;
    if ((rc = own_from(&s0, &h0.a, TSR_F64)) < 0) return rc;
    if ((rc = own_from(&s1, &h1.a, TSR_F64)) < 0) { own_free(&s0); return rc; }
    const int64_t n0 = tsr_shape_size(s0.a.ndim, s0.a.shape), n1 = tsr_shape_size(s1.a.ndim, s1.a.shape);
    double *a0 = (double *)s0.a.data, *a1 = (double *)s1.a.data;
    for (int64_t i = 0; i < n0; i++) if (a0[i] == 0) rc = TSR_EARG;
    for (int64_t i = 0; i < n1; i++) if (a1[i] == 0) rc = TSR_EARG;
    if (rc < 0) { own_free(&s0); own_free(&s1); fn_set_error("Geometric sequence cannot include zero"); return rc; }
    if (num < 0) { own_free(&s0); own_free(&s1); fn_set_error("Number of samples, %lld, must be non-negative.", (long long)num); return TSR_EARG; }
    /* out_sign = sign(start); start /= out_sign; stop = stop / out_sign (broadcast) */
    int32_t nd;
    int64_t sh[TSR_MAXDIM];
    if ((rc = bshape(s0.a.ndim, s0.a.shape, s1.a.ndim, s1.a.shape, &nd, sh)) < 0) { own_free(&s0); own_free(&s1); return rc; }
    own sg;
    if ((rc = own_new(&sg, TSR_F64, s0.a.ndim, s0.a.shape)) < 0) { own_free(&s0); own_free(&s1); return rc; }
    double *sgn = (double *)sg.a.data;
    for (int64_t i = 0; i < n0; i++) { sgn[i] = sign_d(a0[i]); a0[i] = a0[i] / sgn[i]; }
    own st2;
    if ((rc = bop(OP_TDIV, &s1.a, 0, &sg.a, 0, &st2)) < 0) { own_free(&s0); own_free(&s1); own_free(&sg); return rc; }
    own l0, l1;
    if ((rc = own_from(&l0, &s0.a, TSR_F64)) < 0 || (rc = own_from(&l1, &st2.a, TSR_F64)) < 0) {
        own_free(&s0); own_free(&s1); own_free(&sg); own_free(&st2); return rc;
    }
    const int64_t nl0 = tsr_shape_size(l0.a.ndim, l0.a.shape), nl1 = tsr_shape_size(l1.a.ndim, l1.a.shape);
    for (int64_t i = 0; i < nl0; i++) ((double *)l0.a.data)[i] = log10(((double *)l0.a.data)[i]);
    for (int64_t i = 0; i < nl1; i++) ((double *)l1.a.data)[i] = log10(((double *)l1.a.data)[i]);
    own y;
    rc = linspace_core(&l0.a, &l1.a, num, endpoint, &y);
    own_free(&l0);
    own_free(&l1);
    if (rc == 0) {
        /* power(10.0, y), then the exact endpoints, then *= out_sign (broadcast over the trailing axes) */
        double *p = (double *)y.a.data;
        const int64_t m = tsr_shape_size(y.a.ndim - 1, y.a.shape + 1);
        for (int64_t i = 0; i < num * m; i++) p[i] = pow(10.0, p[i]);
        tsr_array A = s0.a, S = st2.a, G = sg.a;
        int64_t ssh[TSR_MAXDIM];
        memcpy(ssh, y.a.shape + 1, sizeof(int64_t) * (size_t)(y.a.ndim - 1));
        if ((rc = v_broadcast(&A, y.a.ndim - 1, ssh)) == 0 && (rc = v_broadcast(&S, y.a.ndim - 1, ssh)) == 0 && (rc = v_broadcast(&G, y.a.ndim - 1, ssh)) == 0) {
            int64_t ix[TSR_MAXDIM] = {0};
            for (int64_t q = 0; q < m; q++) {
                int64_t oa = 0, os = 0, og = 0;
                for (int d = 0; d < y.a.ndim - 1; d++) { oa += ix[d] * A.strides[d]; os += ix[d] * S.strides[d]; og += ix[d] * G.strides[d]; }
                if (num > 0) p[q] = rd_f64(elem0(&A) + oa, TSR_F64);
                if (num > 1 && endpoint) p[(num - 1) * m + q] = rd_f64(elem0(&S) + os, TSR_F64);
                const double g = rd_f64(elem0(&G) + og, TSR_F64);
                for (int64_t i = 0; i < num; i++) p[i * m + q] *= g;
                for (int d = y.a.ndim - 2; d >= 0; d--) { if (++ix[d] < ssh[d]) break; ix[d] = 0; }
            }
        }
    }
    own_free(&s0); own_free(&s1); own_free(&sg); own_free(&st2);
    if (rc < 0) { own_free(&y); return rc; }
    if (is_intdt(dt)) {                                  /* result.astype(int): truncation (no floor here) */
        int ax;
        if ((rc = norm_axis(axis, y.a.ndim, &ax)) < 0) { own_free(&y); return rc; }
        tsr_array v = y.a;
        if (ax != 0) {
            const int64_t s = v.shape[0], t = v.strides[0];
            for (int d = 0; d < ax; d++) { v.shape[d] = v.shape[d + 1]; v.strides[d] = v.strides[d + 1]; }
            v.shape[ax] = s; v.strides[ax] = t;
        }
        rc = emit(&res[0], &v, dt);
        own_free(&y);
        return rc;
    }
    return finish_seq(&res[0], &y, axis, dt);
}

/* vander(x, N=None, increasing=False) */
static int r_vander(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr hx;
    int rc = get_arr(argk(args, nargs, 0), &hx, "x");
    if (rc < 0) return rc;
    if (hx.a.ndim != 1) { fn_set_error("x must be a one-dimensional array or sequence."); return TSR_EARG; }
    const int64_t m = hx.a.shape[0];
    int64_t N = m;
    if (!is_none(argk(args, nargs, 1)) && (rc = int_param(argk(args, nargs, 1), "N", &N)) < 0) return rc;
    const tsr_arg *ia = argk(args, nargs, 2);
    const int inc = !is_none(ia) && ia->num != 0;
    if (N < 0) { fn_set_error("negative dimensions are not allowed"); return TSR_EARG; }
    if (hx.a.dtype == TSR_C128) { fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    const int dt = promote2(hx.a.dtype, TSR_I64);
    const int64_t sh[2] = {m, N};
    own z;
    if ((rc = own_new(&z, dt, 2, sh)) < 0) return rc;
    const int64_t zs = tsr_itemsize(dt);
    for (int64_t i = 0; i < m; i++) {
        const char *px = elem0(&hx.a) + i * hx.a.strides[0];
        unsigned char acc[16], xv[16];
        const int64_t one = 1, zz = 0;
        tsr_copy(hx.a.dtype, dt, 1, &one, px, &zz, xv, &zz);
        for (int64_t j = 0; j < N; j++) {
            const int64_t col = inc ? j : N - 1 - j;
            char *q = (char *)z.a.data + (i * N + col) * zs;
            if (j == 0) {
                if (dt == TSR_F64) { const double o = 1.0; memcpy(acc, &o, 8); }
                else if (dt == TSR_F32) { const float o = 1.0f; memcpy(acc, &o, 4); }
                else { const int64_t o = 1; memcpy(acc, &o, 8); }
            } else if (j == 1) memcpy(acc, xv, 16);
            else { unsigned char t[16]; elem_op(OP_MUL, dt, (char *)acc, dt, (char *)xv, dt, (char *)t); memcpy(acc, t, 16); }
            memcpy(q, acc, (size_t)zs);
        }
    }
    return own_emit(&res[0], &z);
}

/* ================================================================ polynomials */

/* atleast_1d view */
static tsr_array at1d(const tsr_array *a)
{
    tsr_array v = *a;
    if (v.ndim == 0) { v.ndim = 1; v.shape[0] = 1; v.strides[0] = 0; }
    return v;
}

/* polyval(p, x) */
static int r_polyval(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr hp, hx;
    int rc = get_arr(argk(args, nargs, 0), &hp, "p");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &hx, "x");
    if (rc < 0) return rc;
    if (hp.a.ndim == 0) { fn_set_error("iteration over a 0-d array"); return TSR_ETYPE; }
    own y;
    if ((rc = own_new(&y, hx.a.dtype, hx.a.ndim, hx.a.shape)) < 0) return rc;   /* zeros_like(x) */
    for (int64_t i = 0; i < hp.a.shape[0]; i++) {
        tsr_array pv = hp.a;
        pv.offset += i * hp.a.strides[0];
        pv.ndim--;
        memmove(pv.shape, pv.shape + 1, sizeof(int64_t) * (size_t)pv.ndim);
        memmove(pv.strides, pv.strides + 1, sizeof(int64_t) * (size_t)pv.ndim);
        own t, u;
        if ((rc = bop(OP_MUL, &y.a, 0, &hx.a, 0, &t)) < 0) { own_free(&y); return rc; }
        own_free(&y);
        if ((rc = bop(OP_ADD, &t.a, 0, &pv, 0, &u)) < 0) { own_free(&t); return rc; }
        own_free(&t);
        y = u;
    }
    return own_emit(&res[0], &y);
}

/* polyadd / polysub (a1, a2) */
static int r_polyaddsub(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int op = *(const int *)ctx;
    harr h1, h2;
    int rc = get_arr(argk(args, nargs, 0), &h1, "a1");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &h2, "a2");
    if (rc < 0) return rc;
    tsr_array a1 = at1d(&h1.a), a2 = at1d(&h2.a);
    const int64_t diff = a2.shape[0] - a1.shape[0];
    own pad, z;
    memset(&pad, 0, sizeof pad);
    tsr_array x = a1, y = a2;
    if (diff != 0) {
        tsr_array *shorter = diff > 0 ? &x : &y;
        if (shorter->ndim != 1) {
            fn_set_error("all the input array dimensions except for the concatenation axis must match exactly");
            return TSR_EARG;
        }
        const int64_t len = shorter->shape[0] + (diff > 0 ? diff : -diff);
        if ((rc = own_new(&pad, shorter->dtype, 1, &len)) < 0) return rc;
        const int64_t off = diff > 0 ? diff : -diff;
        if (shorter->shape[0] > 0) copy_view(shorter, shorter->dtype, (char *)pad.a.data + off * pad.a.strides[0], pad.a.strides);
        *shorter = pad.a;
    }
    rc = bop(op, &x, 0, &y, 0, &z);
    own_free(&pad);
    if (rc < 0) return rc;
    return own_emit(&res[0], &z);
}
static const int PADD = OP_ADD, PSUB = OP_SUB;

/* poly1d(c).coeffs: 1-D, leading zeros trimmed, [0] when nothing is left */
static int poly_coeffs(const tsr_array *a, tsr_array *out)
{
    tsr_array v = at1d(a);
    if (v.ndim > 1) { fn_set_error("Polynomial must be 1d only."); return TSR_EARG; }
    int64_t k = 0;
    while (k < v.shape[0] && rd_f64(elem0(&v) + k * v.strides[0], v.dtype) == 0 && !(v.dtype == TSR_C128)) k++;
    v.offset += k * v.strides[0];
    v.shape[0] -= k;
    *out = v;
    return TSR_OK;
}

/* numpy.convolve(a, v) of 1-D arrays, full mode, in the promoted dtype */
static int convolve_full(const tsr_array *a0, const tsr_array *v0, own *z)
{
    tsr_array a = *a0, v = *v0;
    if (a.shape[0] == 0) { fn_set_error("a cannot be empty"); return TSR_EARG; }
    if (v.shape[0] == 0) { fn_set_error("v cannot be empty"); return TSR_EARG; }
    if (v.shape[0] > a.shape[0]) { tsr_array t = a; a = v; v = t; }
    const int dt = promote2(a.dtype, v.dtype);
    if (dt == TSR_C128) { fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    const int64_t n1 = a.shape[0], n2 = v.shape[0], len = n1 + n2 - 1;
    int rc = own_new(z, dt, 1, &len);
    if (rc < 0) return rc;
    const int64_t zs = tsr_itemsize(dt);
    /* correlate(a, v[::-1], 'full'): out[k] = sum_j a[j] * v[k - j] over the overlap, j ascending */
    for (int64_t k = 0; k < len; k++) {
        const int64_t jlo = k - (n2 - 1) > 0 ? k - (n2 - 1) : 0, jhi = k < n1 - 1 ? k : n1 - 1;
        char *q = (char *)z->a.data + k * zs;
        if (dt == TSR_F64 || dt == TSR_F32) {
            double s = 0;
            for (int64_t j = jlo; j <= jhi; j++) s += rd_f64(elem0(&a) + j * a.strides[0], a.dtype) * rd_f64(elem0(&v) + (k - j) * v.strides[0], v.dtype);
            if (dt == TSR_F64) memcpy(q, &s, 8); else { const float f = (float)s; memcpy(q, &f, 4); }
        } else if (dt == TSR_BOOL) {
            int s = 0;
            for (int64_t j = jlo; j <= jhi && !s; j++) s = rd_i64(elem0(&a) + j * a.strides[0], a.dtype) && rd_i64(elem0(&v) + (k - j) * v.strides[0], v.dtype);
            *(uint8_t *)q = (uint8_t)s;
        } else {
            uint64_t s = 0;
            for (int64_t j = jlo; j <= jhi; j++) s += (uint64_t)rd_i64(elem0(&a) + j * a.strides[0], a.dtype) * (uint64_t)rd_i64(elem0(&v) + (k - j) * v.strides[0], v.dtype);
            if (dt == TSR_I64) { const int64_t w = (int64_t)s; memcpy(q, &w, 8); }
            else if (dt == TSR_I32) { const int32_t w = (int32_t)(uint32_t)s; memcpy(q, &w, 4); }
            else *(uint8_t *)q = (uint8_t)s;
        }
    }
    return TSR_OK;
}

/* polymul(a1, a2) */
static int r_polymul(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h1, h2;
    int rc = get_arr(argk(args, nargs, 0), &h1, "a1");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &h2, "a2");
    if (rc < 0) return rc;
    tsr_array c1, c2;
    if ((rc = poly_coeffs(&h1.a, &c1)) < 0 || (rc = poly_coeffs(&h2.a, &c2)) < 0) return rc;
    harr z1, z2;
    if (c1.shape[0] == 0) { memset(&z1, 0, sizeof z1); z1.a.data = z1.s; z1.a.dtype = c1.dtype; z1.a.ndim = 1; z1.a.shape[0] = 1; z1.a.strides[0] = 0; c1 = z1.a; }
    if (c2.shape[0] == 0) { memset(&z2, 0, sizeof z2); z2.a.data = z2.s; z2.a.dtype = c2.dtype; z2.a.ndim = 1; z2.a.shape[0] = 1; z2.a.strides[0] = 0; c2 = z2.a; }
    own z;
    if ((rc = convolve_full(&c1, &c2, &z)) < 0) return rc;
    return own_emit(&res[0], &z);
}

/* polyder(p, m=1) */
static int r_polyder(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr hp;
    int rc = get_arr(argk(args, nargs, 0), &hp, "p");
    if (rc < 0) return rc;
    int64_t m = 1;
    if (!is_none(argk(args, nargs, 1)) && (rc = int_trunc(argk(args, nargs, 1), "m", &m)) < 0) return rc;
    if (m < 0) { fn_set_error("Order of derivative must be positive (see polyint)"); return TSR_EARG; }
    /* numpy.polyder recurses m times: Python's recursion limit (1000) stops it near m = 1000 */
    if (m >= 1000) { fn_set_error("maximum recursion depth exceeded (numpy.polyder recurses once per order)"); return TSR_EARG; }
    if (hp.a.ndim == 0) { fn_set_error("len() of unsized object"); return TSR_ETYPE; }
    own cur;
    if ((rc = own_from(&cur, &hp.a, -1)) < 0) return rc;
    for (int64_t k = 0; k < m; k++) {
        /* y = p[:-1] * arange(n, 0, -1) */
        const int64_t n = cur.a.shape[0] - 1;
        tsr_array head = cur.a;
        v_slice(&head, 0, 0, n > 0 ? n : 0);
        own ar, y;
        const int64_t len = n > 0 ? n : 0;
        if ((rc = own_new(&ar, TSR_I64, 1, &len)) < 0) { own_free(&cur); return rc; }
        for (int64_t i = 0; i < len; i++) ((int64_t *)ar.a.data)[i] = n - i;
        rc = bop(OP_MUL, &head, 0, &ar.a, 0, &y);
        own_free(&ar);
        own_free(&cur);
        if (rc < 0) return rc;
        cur = y;
        if (cur.a.ndim == 1 && cur.a.shape[0] == 0 && k + 1 < m) {
            /* p[:-1] of an empty array is empty; numpy keeps going with len -1 -> arange(-1, 0, -1) = [] */
        }
    }
    return own_emit(&res[0], &cur);
}

/* polyint(p, m=1, k=None) */
static int r_polyint(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr hp;
    int rc = get_arr(argk(args, nargs, 0), &hp, "p");
    if (rc < 0) return rc;
    int64_t m = 1;
    if (!is_none(argk(args, nargs, 1)) && (rc = int_trunc(argk(args, nargs, 1), "m", &m)) < 0) return rc;
    if (m < 0) { fn_set_error("Order of integral must be positive (see polyder)"); return TSR_EARG; }
    /* numpy.polyint recurses m times: Python's recursion limit (1000) stops it near m = 1000 */
    if (m >= 1000) { fn_set_error("maximum recursion depth exceeded (numpy.polyint recurses once per order)"); return TSR_EARG; }
    /* k: None -> zeros(m); a scalar or length-1 -> repeated when m > 1; else at least m values */
    const tsr_arg *ka = argk(args, nargs, 2);
    double *kv = NULL;
    int64_t nk = m, kb = (m + 2) * 8 + 64;
    kv = (double *)tsr_calloc(kb);
    if (!kv) return TSR_ENOMEM;
    int kdt = TSR_F64;
    if (!is_none(ka)) {
        harr hk;
        if ((rc = get_arr(ka, &hk, "k")) < 0) { tsr_free(kv, kb); return rc; }
        tsr_array k1 = at1d(&hk.a);
        kdt = k1.dtype;
        const int64_t len = k1.shape[0];
        if (len == 1 && m > 1) { const double v = rd_f64(elem0(&k1), k1.dtype); for (int64_t i = 0; i < m; i++) kv[i] = v; kdt = TSR_F64; }
        else {
            if (len < m) { tsr_free(kv, kb); fn_set_error("k must be a scalar or a rank-1 array of length 1 or >m."); return TSR_EARG; }
            tsr_free(kv, kb);
            kb = (len + 2) * 8 + 64;
            kv = (double *)tsr_calloc(kb);
            if (!kv) return TSR_ENOMEM;
            for (int64_t i = 0; i < len; i++) kv[i] = rd_f64(elem0(&k1) + i * k1.strides[0], k1.dtype);
            nk = len;
        }
    }
    (void)nk;
    (void)kdt;
    if (hp.a.ndim == 0) { tsr_free(kv, kb); fn_set_error("len() of unsized object"); return TSR_ETYPE; }
    own cur;
    if (m == 0) { tsr_free(kv, kb); return emit(&res[0], &hp.a, -1); }
    if ((rc = own_from(&cur, &hp.a, -1)) < 0) { tsr_free(kv, kb); return rc; }
    for (int64_t it = 0; it < m; it++) {
        /* y = concatenate((p / arange(len(p), 0, -1), [k[it]])) */
        const int64_t n = cur.a.shape[0];
        own ar, q, y;
        if ((rc = own_new(&ar, TSR_I64, 1, &n)) < 0) break;
        for (int64_t i = 0; i < n; i++) ((int64_t *)ar.a.data)[i] = n - i;
        rc = bop(OP_TDIV, &cur.a, 0, &ar.a, 0, &q);
        own_free(&ar);
        if (rc < 0) break;
        const int64_t n1 = n + 1;
        const int ydt = promote2(q.a.dtype, TSR_F64);
        if ((rc = own_new(&y, ydt, 1, &n1)) < 0) { own_free(&q); break; }
        if (n > 0) copy_view(&q.a, ydt, y.a.data, y.a.strides);
        if (ydt == TSR_F64) ((double *)y.a.data)[n] = kv[it]; else ((float *)y.a.data)[n] = (float)kv[it];
        own_free(&q);
        own_free(&cur);
        cur = y;
    }
    tsr_free(kv, kb);
    if (rc < 0) { own_free(&cur); return rc; }
    return own_emit(&res[0], &cur);
}

/* polydiv(u, v) -> [q, r] */
static int r_polydiv(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr hu, hv;
    int rc = get_arr(argk(args, nargs, 0), &hu, "u");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &hv, "v");
    if (rc < 0) return rc;
    tsr_array u1 = at1d(&hu.a), v1 = at1d(&hv.a);
    if (u1.ndim != 1 || v1.ndim != 1) { fn_set_error("polydiv takes 1-D coefficient arrays"); return TSR_EARG; }
    if (u1.dtype == TSR_C128 || v1.dtype == TSR_C128) { fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    if (u1.shape[0] == 0 || v1.shape[0] == 0) { fn_set_error("index 0 is out of bounds for axis 0 with size 0"); return TSR_EARG; }
    /* u + 0.0, v + 0.0: float64 (float32 stays float32) */
    const int du = u1.dtype == TSR_F32 ? TSR_F32 : TSR_F64, dv = v1.dtype == TSR_F32 ? TSR_F32 : TSR_F64;
    const int dt = promote2(du, dv);
    const int64_t M = u1.shape[0] - 1, N = v1.shape[0] - 1;
    const int64_t nq = M - N + 1 > 1 ? M - N + 1 : 1;
    const int64_t wb = (M + N + nq + 4) * 8 + 64;
    double *w = (double *)tsr_calloc(wb);
    if (!w) return TSR_ENOMEM;
    double *r = w, *v = w + M + 2, *q = v + N + 2;
    for (int64_t i = 0; i <= M; i++) r[i] = rd_f64(elem0(&u1) + i * u1.strides[0], u1.dtype);
    for (int64_t i = 0; i <= N; i++) v[i] = rd_f64(elem0(&v1) + i * v1.strides[0], v1.dtype);
    const double scale = 1. / v[0];
    for (int64_t k = 0; k < M - N + 1; k++) {
        const double d = scale * r[k];
        q[k] = d;
        for (int64_t j = 0; j <= N; j++) r[k + j] -= d * v[j];
    }
    int64_t start = 0, rlen = M + 1;
    /* while allclose(r[0], 0, rtol=1e-14) and len > 1: |r0| <= 1e-8 + 1e-14 * 0 */
    while (rlen > 1 && (fabs(r[start]) <= 1e-8 || (r[start] != r[start] && 0))) { start++; rlen--; }
    tsr_result *items = fn_result_seq(&res[0], 2);
    if (!items) { tsr_free(w, wb); return TSR_ENOMEM; }
    void *pq = fn_result_array(&items[0], dt, 1, &nq), *pr = pq ? fn_result_array(&items[1], dt, 1, &rlen) : NULL;
    if (!pq || !pr) { tsr_free(w, wb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < nq; i++) { if (dt == TSR_F64) ((double *)pq)[i] = q[i]; else ((float *)pq)[i] = (float)q[i]; }
    for (int64_t i = 0; i < rlen; i++) { if (dt == TSR_F64) ((double *)pr)[i] = r[start + i]; else ((float *)pr)[i] = (float)r[start + i]; }
    tsr_free(w, wb);
    return TSR_OK;
}

/* ================================================================ windows, sinc, i0 */

/* M as numpy does: values = np.array([0.0, M]) -> a float */
static int window_m(const tsr_arg *a, double *M)
{
    if (a->kind == 1) { *M = (a->flags & 1) ? (double)a->ival : a->num; return TSR_OK; }
    if (a->kind == 4) { *M = a->num != 0; return TSR_OK; }
    if (a->kind == 3 && a->arr.ndim == 0 && a->arr.dtype != TSR_C128) { *M = rd_f64(elem0(&a->arr), a->arr.dtype); return TSR_OK; }
    fn_set_error("M must be a number");
    return TSR_EARG;
}

/* numpy.arange(start, stop, step) of floats: length ceil((stop - start) / step), values start + i * delta
   with delta = (start + step) - start (numpy's fill) */
static int64_t arange_len(double start, double stop, double step)
{
    const double len = ceil((stop - start) / step);
    if (!(len > 0)) return 0;
    if (len > 1e15) return -1;
    return (int64_t)len;
}

enum { W_BARTLETT, W_BLACKMAN, W_HAMMING, W_HANNING };

static int r_window(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int kind = *(const int *)ctx;
    double M;
    int rc = window_m(argk(args, nargs, 0), &M);
    if (rc < 0) return rc;
    own z;
    if (M < 1 || M != M) { const int64_t zero = 0; if ((rc = own_new(&z, TSR_F64, 1, &zero)) < 0) return rc; return own_emit(&res[0], &z); }
    if (M == 1) { const int64_t one = 1; if ((rc = own_new(&z, TSR_F64, 1, &one)) < 0) return rc; ((double *)z.a.data)[0] = 1.0; return own_emit(&res[0], &z); }
    const double start = 1 - M;
    const int64_t n = arange_len(start, M, 2.0);
    if (n < 0) { fn_set_error("Maximum allowed size exceeded"); return TSR_EARG; }
    if ((rc = own_new(&z, TSR_F64, 1, &n)) < 0) return rc;
    double *p = (double *)z.a.data;
    const double delta = (start + 2.0) - start;
    for (int64_t i = 0; i < n; i++) {
        const double x = start + (double)i * delta;
        switch (kind) {
        case W_BARTLETT: p[i] = x <= 0 ? 1 + x / (M - 1) : 1 - x / (M - 1); break;
        case W_BLACKMAN: p[i] = 0.42 + 0.5 * cos(M_PI * x / (M - 1)) + 0.08 * cos(2.0 * M_PI * x / (M - 1)); break;
        case W_HAMMING: p[i] = 0.54 + 0.46 * cos(M_PI * x / (M - 1)); break;
        default: p[i] = 0.5 + 0.5 * cos(M_PI * x / (M - 1)); break;
        }
    }
    return own_emit(&res[0], &z);
}
static const int WB = W_BARTLETT, WBL = W_BLACKMAN, WHM = W_HAMMING, WHN = W_HANNING;

static const double I0A[] = {
    -4.41534164647933937950E-18, 3.33079451882223809783E-17, -2.43127984654795469359E-16, 1.71539128555513303061E-15,
    -1.16853328779934516808E-14, 7.67618549860493561688E-14, -4.85644678311192946090E-13, 2.95505266312963983461E-12,
    -1.72682629144155570723E-11, 9.67580903537323691224E-11, -5.18979560163526290666E-10, 2.65982372468238665035E-9,
    -1.30002500998624804212E-8, 6.04699502254191894932E-8, -2.67079385394061173391E-7, 1.11738753912010371815E-6,
    -4.41673835845875056359E-6, 1.64484480707288970893E-5, -5.75419501008210370398E-5, 1.88502885095841655729E-4,
    -5.76375574538582365885E-4, 1.63947561694133579842E-3, -4.32430999505057594430E-3, 1.05464603945949983183E-2,
    -2.37374148058994688156E-2, 4.93052842396707084878E-2, -9.49010970480476444210E-2, 1.71620901522208775349E-1,
    -3.04682672343198398683E-1, 6.76795274409476084995E-1};
static const double I0B[] = {
    -7.23318048787475395456E-18, -4.83050448594418207126E-18, 4.46562142029675999901E-17, 3.46122286769746109310E-17,
    -2.82762398051658348494E-16, -3.42548561967721913462E-16, 1.77256013305652638360E-15, 3.81168066935262242075E-15,
    -9.55484669882830764870E-15, -4.15056934728722208663E-14, 1.54008621752140982691E-14, 3.85277838274214270114E-13,
    7.18012445138366623367E-13, -1.79417853150680611778E-12, -1.32158118404477131188E-11, -3.14991652796324136454E-11,
    1.18891471078464383424E-11, 4.94060238822496958910E-10, 3.39623202570838634515E-9, 2.26666899049817806459E-8,
    2.04891858946906374183E-7, 2.89137052083475648297E-6, 6.88975834691682398426E-5, 3.36911647825569408990E-3,
    8.04490411014108831608E-1};

static double chbevl(double x, const double *vals, int n)
{
    double b0 = vals[0], b1 = 0.0, b2 = 0.0;
    for (int i = 1; i < n; i++) { b2 = b1; b1 = b0; b0 = x * b1 - b2 + vals[i]; }
    return 0.5 * (b0 - b2);
}

/* numpy.i0 of |x|: piecewise(x, [x <= 8], [_i0_1, _i0_2]) */
static double np_i0(double x)
{
    x = fabs(x);
    if (x <= 8.0) return exp(x) * chbevl(x / 2.0 - 2, I0A, 30);
    return exp(x) * chbevl(32.0 / x - 2.0, I0B, 25) / sqrt(x);
}

static int r_i0(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "x");
    if (rc < 0) return rc;
    if (h.a.dtype == TSR_C128) { fn_set_error("i0 not supported for complex values"); return TSR_ETYPE; }
    own z;
    const int dt = h.a.dtype == TSR_F32 ? TSR_F32 : TSR_F64;
    if ((rc = own_from(&z, &h.a, TSR_F64)) < 0) return rc;
    const int64_t n = tsr_shape_size(z.a.ndim, z.a.shape);
    double *p = (double *)z.a.data;
    for (int64_t i = 0; i < n; i++) p[i] = np_i0(p[i]);
    if (dt == TSR_F32) { rc = emit(&res[0], &z.a, TSR_F32); own_free(&z); return rc; }
    return own_emit(&res[0], &z);
}

/* kaiser(M, beta) */
static int r_kaiser(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    double M, beta;
    int rc = window_m(argk(args, nargs, 0), &M);
    if (rc == 0) rc = num_param(argk(args, nargs, 1), "beta", &beta);
    if (rc < 0) return rc;
    own z;
    if (M == 1) { const int64_t one = 1; if ((rc = own_new(&z, TSR_F64, 1, &one)) < 0) return rc; ((double *)z.a.data)[0] = 1.0; return own_emit(&res[0], &z); }
    const int64_t n = M == M ? arange_len(0, M, 1.0) : -2;
    if (n == -2) { fn_set_error("arange: cannot compute length"); return TSR_EARG; }
    if (n < 0) { fn_set_error("Maximum allowed size exceeded"); return TSR_EARG; }
    if ((rc = own_new(&z, TSR_F64, 1, &n)) < 0) return rc;
    double *p = (double *)z.a.data;
    const double alpha = (M - 1) / 2.0, den = np_i0(beta);
    for (int64_t i = 0; i < n; i++) {
        const double t = ((double)i - alpha) / alpha;
        p[i] = np_i0(beta * sqrt(1 - t * t)) / den;
    }
    return own_emit(&res[0], &z);
}

/* sinc(x): y = pi * x; where(y, y, eps); sin(y) / y */
static int r_sinc(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "x");
    if (rc < 0) return rc;
    if (h.a.dtype == TSR_C128) { fn_set_error("complex arrays are not supported here"); return TSR_ETYPE; }
    const int f32 = h.a.dtype == TSR_F32;
    own z;
    if ((rc = own_from(&z, &h.a, f32 ? TSR_F32 : TSR_F64)) < 0) return rc;
    const int64_t n = tsr_shape_size(z.a.ndim, z.a.shape);
    if (f32) {
        float *p = (float *)z.a.data;
        for (int64_t i = 0; i < n; i++) {
            float y = (float)M_PI * p[i];
            if (y == 0) y = 1.1920929e-07f;
            p[i] = (float)sin((double)y) / y;
        }
    } else {
        double *p = (double *)z.a.data;
        for (int64_t i = 0; i < n; i++) {
            double y = M_PI * p[i];
            if (y == 0) y = 2.220446049250313e-16;
            p[i] = sin(y) / y;
        }
    }
    return own_emit(&res[0], &z);
}

/* ================================================================ nan_to_num and the type predicates */

/* nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None) */
static int r_nan_to_num(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "x");
    if (rc < 0) return rc;
    double vn = 0.0, vp, vm;
    const int dt = h.a.dtype;
    if (!is_none(argk(args, nargs, 2)) && (rc = num_param(argk(args, nargs, 2), "nan", &vn)) < 0) return rc;
    const double fmax = dt == TSR_F32 ? 3.4028234663852886e38 : 1.7976931348623157e308;
    vp = fmax;
    vm = -fmax;
    if (!is_none(argk(args, nargs, 3)) && (rc = num_param(argk(args, nargs, 3), "posinf", &vp)) < 0) return rc;
    if (!is_none(argk(args, nargs, 4)) && (rc = num_param(argk(args, nargs, 4), "neginf", &vm)) < 0) return rc;
    own z;
    if ((rc = own_from(&z, &h.a, -1)) < 0) return rc;
    const int64_t n = tsr_shape_size(z.a.ndim, z.a.shape);
    if (dt == TSR_F64 || dt == TSR_C128) {
        double *p = (double *)z.a.data;
        const int64_t m = dt == TSR_C128 ? 2 * n : n;
        for (int64_t i = 0; i < m; i++) {
            if (p[i] != p[i]) p[i] = vn;
            else if (p[i] == INFINITY) p[i] = vp;
            else if (p[i] == -INFINITY) p[i] = vm;
        }
    } else if (dt == TSR_F32) {
        float *p = (float *)z.a.data;
        for (int64_t i = 0; i < n; i++) {
            if (p[i] != p[i]) p[i] = (float)vn;
            else if (p[i] == INFINITY) p[i] = (float)vp;
            else if (p[i] == -INFINITY) p[i] = (float)vm;
        }
    }
    return own_emit(&res[0], &z);
}

/* isposinf / isneginf (x) */
static int r_isinf_sign(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int neg = *(const int *)ctx;
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "x");
    if (rc < 0) return rc;
    if (h.a.dtype == TSR_C128) { fn_set_error("This operation is not supported for complex128 values because it would be ambiguous."); return TSR_ETYPE; }
    own z;
    if ((rc = own_new(&z, TSR_BOOL, h.a.ndim, h.a.shape)) < 0) return rc;
    const int64_t n = tsr_shape_size(h.a.ndim, h.a.shape);
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t o = 0;
        for (int d = 0; d < h.a.ndim; d++) o += ix[d] * h.a.strides[d];
        const double v = rd_f64(elem0(&h.a) + o, h.a.dtype);
        ((uint8_t *)z.a.data)[q] = (uint8_t)(isinf(v) && (neg ? signbit(v) : !signbit(v)));
        for (int d = h.a.ndim - 1; d >= 0; d--) { if (++ix[d] < h.a.shape[d]) break; ix[d] = 0; }
    }
    return own_emit(&res[0], &z);
}
static const int POS = 0, NEG = 1;

/* iscomplex / isreal (x): element-wise on the imaginary part; iscomplexobj / isrealobj (x): the dtype */
static int r_complex_pred(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int kind = *(const int *)ctx;                  /* 0 iscomplex, 1 isreal, 2 iscomplexobj, 3 isrealobj */
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "x");
    if (rc < 0) return rc;
    const int cplx = h.a.dtype == TSR_C128;
    if (kind >= 2) {
        memset(&res[0], 0, sizeof res[0]);
        res[0].kind = 4;
        res[0].num = kind == 2 ? cplx : !cplx;
        return TSR_OK;
    }
    own z;
    if ((rc = own_new(&z, TSR_BOOL, h.a.ndim, h.a.shape)) < 0) return rc;
    const int64_t n = tsr_shape_size(h.a.ndim, h.a.shape);
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < n; q++) {
        double im = 0.0;
        if (cplx) {
            int64_t o = 0;
            for (int d = 0; d < h.a.ndim; d++) o += ix[d] * h.a.strides[d];
            memcpy(&im, elem0(&h.a) + o + 8, 8);
        }
        ((uint8_t *)z.a.data)[q] = (uint8_t)(kind == 0 ? (cplx && im != 0) : (im == 0));
        for (int d = h.a.ndim - 1; d >= 0; d--) { if (++ix[d] < h.a.shape[d]) break; ix[d] = 0; }
    }
    return own_emit(&res[0], &z);
}
static const int P_ISCOMPLEX = 0, P_ISREAL = 1, P_ISCOMPLEXOBJ = 2, P_ISREALOBJ = 3;

/* real_if_close(a, tol=100) */
static int r_real_if_close(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    if (h.a.dtype != TSR_C128) return emit(&res[0], &h.a, -1);
    double tol = 100;
    if (!is_none(argk(args, nargs, 1)) && (rc = num_param(argk(args, nargs, 1), "tol", &tol)) < 0) return rc;
    if (tol > 1) tol = 2.220446049250313e-16 * tol;
    const int64_t n = tsr_shape_size(h.a.ndim, h.a.shape);
    int64_t ix[TSR_MAXDIM] = {0};
    int all = 1;
    for (int64_t q = 0; q < n && all; q++) {
        int64_t o = 0;
        for (int d = 0; d < h.a.ndim; d++) o += ix[d] * h.a.strides[d];
        double im;
        memcpy(&im, elem0(&h.a) + o + 8, 8);
        if (!(fabs(im) < tol)) all = 0;
        for (int d = h.a.ndim - 1; d >= 0; d--) { if (++ix[d] < h.a.shape[d]) break; ix[d] = 0; }
    }
    return emit(&res[0], &h.a, all ? TSR_F64 : -1);
}

/* ================================================================ pad */

/* copy src (broadcast to dst's shape) into the strided region dst, casting to dst's dtype */
static int assign_bc(const tsr_array *dst, const tsr_array *src)
{
    if (tsr_shape_size(dst->ndim, dst->shape) == 0) return TSR_OK;
    tsr_array s = *src;
    int rc = v_broadcast(&s, dst->ndim, dst->shape);
    if (rc < 0) return rc;
    if (dst->ndim == 0) {
        const int64_t one = 1, z = 0;
        return tsr_copy(s.dtype, dst->dtype, 1, &one, elem0(&s), &z, elem0(dst), &z);
    }
    return tsr_copy(s.dtype, dst->dtype, dst->ndim, dst->shape, elem0(&s), s.strides, elem0(dst), dst->strides);
}

/* numpy.lib._arraypad_impl._as_pairs: an (ndim, 2) view of x (size 1 -> both sides of every axis; size 2
   (not shaped (2, 1)) -> (before, after) for every axis; otherwise broadcast to (ndim, 2)) */
static int as_pairs(const tsr_array *x, int nd, tsr_array *out)
{
    const int64_t n = tsr_shape_size(x->ndim, x->shape);
    tsr_array v = *x;
    const int64_t sh[2] = {nd, 2};
    if (x->ndim < 3 && n == 1) {
        v.ndim = 2;
        v.shape[0] = nd; v.shape[1] = 2;
        v.strides[0] = v.strides[1] = 0;
        *out = v;
        return TSR_OK;
    }
    if (x->ndim < 3 && n == 2 && !(x->ndim == 2 && x->shape[0] == 2 && x->shape[1] == 1)) {
        /* ravel: two elements in C order */
        int64_t s = 0;
        if (x->ndim == 1) s = x->strides[0];
        else s = x->shape[1] == 2 ? x->strides[1] : x->strides[0];
        v.ndim = 2;
        v.shape[0] = nd; v.shape[1] = 2;
        v.strides[0] = 0; v.strides[1] = s;
        *out = v;
        return TSR_OK;
    }
    if (v_broadcast(&v, 2, sh) < 0) {
        fn_set_error("operands could not be broadcast together with remapped shapes [original->remapped]: (%lld,) and requested shape (%d,2)", (long long)(x->ndim ? x->shape[0] : 0), nd);
        return TSR_EARG;
    }
    *out = v;
    return TSR_OK;
}

/* integer pairs (pad widths, stat lengths): np.round(x).astype(intp); negative raises */
static int as_index_pairs(const tsr_arg *a, int nd, int64_t (*pairs)[2], const char *what)
{
    harr h;
    int rc = get_arr(a, &h, what);
    if (rc < 0) return rc;
    tsr_array v;
    if ((rc = as_pairs(&h.a, nd, &v)) < 0) return rc;
    for (int d = 0; d < nd; d++)
        for (int s = 0; s < 2; s++) {
            const char *p = elem0(&v) + d * v.strides[0] + s * v.strides[1];
            int64_t w;
            if (is_intlike(v.dtype)) w = rd_i64(p, v.dtype);
            else {
                const double x = nearbyint(rd_f64(p, v.dtype));
                if (!(x > -9.2e18 && x < 9.2e18)) { fn_set_error("cannot convert float NaN to integer"); return TSR_EARG; }
                w = (int64_t)x;
            }
            if (w < 0) { fn_set_error("index can't contain negative values"); return TSR_EARG; }
            pairs[d][s] = w;
        }
    return TSR_OK;
}

static tsr_array slice_at(const tsr_array *a, int ax, int64_t start, int64_t stop)
{
    tsr_array v = *a;
    v_slice(&v, ax, start, stop);
    return v;
}

/* the statistic of a chunk along ax, keepdims, into an owned array of padded's shape with length 1 on ax */
static int pad_stat(const tsr_array *chunk, int ax, int mode, int dt, own *out)
{
    int64_t sh[TSR_MAXDIM];
    memcpy(sh, chunk->shape, sizeof(int64_t) * (size_t)chunk->ndim);
    sh[ax] = 1;
    const int sdt = mode == 2 || mode == 3 ? (dt == TSR_F32 ? TSR_F32 : TSR_F64) : dt;   /* mean / median: float */
    int rc = own_new(out, sdt, chunk->ndim, sh);
    if (rc < 0) return rc;
    const int nd = chunk->ndim;
    const int64_t len = chunk->shape[ax];
    int64_t osh[TSR_MAXDIM];
    int k = 0;
    for (int d = 0; d < nd; d++) if (d != ax) osh[k++] = chunk->shape[d];
    const int64_t nl = tsr_shape_size(nd - 1, osh);
    const int64_t lb = (len + 1) * 8 + 64;
    double *lane = (double *)tsr_alloc(lb);
    if (!lane) { own_free(out); return TSR_ENOMEM; }
    int inner = 1;
    for (int d = ax + 1; d < nd; d++) if (chunk->shape[d] != 1) inner = 0;
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t l = 0; l < nl; l++) {
        int64_t base = 0, ob = 0, kk = 0;
        for (int d = 0; d < nd; d++) if (d != ax) { base += ix[kk] * chunk->strides[d]; ob += ix[kk] * out->a.strides[d]; kk++; }
        char *q = (char *)out->a.data + ob;
        if (mode == 0 || mode == 1) {                    /* maximum / minimum in the array dtype (NaN propagates) */
            const char *best = elem0(chunk) + base;
            for (int64_t i = 1; i < len; i++) {
                const char *p = elem0(chunk) + base + i * chunk->strides[ax];
                if (is_floatdt(dt)) {
                    const double x = rd_f64(p, dt), b = rd_f64(best, dt);
                    if (b != b) break;
                    if (x != x || (mode == 0 ? x > b : x < b)) best = p;
                } else {
                    const int64_t x = rd_i64(p, dt), b = rd_i64(best, dt);
                    if (mode == 0 ? x > b : x < b) best = p;
                }
            }
            memcpy(q, best, (size_t)tsr_itemsize(dt));
        } else {
            for (int64_t i = 0; i < len; i++) lane[i] = rd_f64(elem0(chunk) + base + i * chunk->strides[ax], dt);
            double v;
            if (mode == 2) v = lane_sum(lane, len, inner) / (double)len;             /* numpy.mean */
            else {                                                                    /* numpy.median */
                int nan = 0;
                for (int64_t i = 0; i < len; i++) if (lane[i] != lane[i]) nan = 1;
                if (nan) v = NAN;
                else if (len == 0) v = NAN;
                else {
                    for (int64_t i = 1; i < len; i++) {                               /* insertion sort: short chunks */
                        const double t = lane[i];
                        int64_t j = i;
                        while (j > 0 && lane[j - 1] > t) { lane[j] = lane[j - 1]; j--; }
                        lane[j] = t;
                    }
                    v = len % 2 ? lane[len / 2] : (lane[len / 2 - 1] + lane[len / 2]) / 2.0;
                }
            }
            if (sdt == TSR_F32) { const float f = (float)v; memcpy(q, &f, 4); } else memcpy(q, &v, 8);
        }
        for (int d = nd - 2; d >= 0; d--) { if (++ix[d] < osh[d]) break; ix[d] = 0; }
    }
    tsr_free(lane, lb);
    if (is_intdt(dt) && sdt != dt) {                   /* _round_if_needed (mean, median): round half to even for integer (not bool) arrays */
        const int64_t n = tsr_shape_size(nd, sh);
        for (int64_t i = 0; i < n; i++) ((double *)out->a.data)[i] = nearbyint(((double *)out->a.data)[i]);
    }
    return TSR_OK;
}

/* pad(array, pad_width, mode='constant', constant_values=None, end_values=None, stat_length=None, reflect_type=None) */
static int r_pad(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "array");
    if (rc < 0) return rc;
    const tsr_array *arr = &h.a;
    const int nd = arr->ndim, dt = arr->dtype;
    /* pad_width must be integral */
    const tsr_arg *pw = argk(args, nargs, 1);
    {
        harr hp;
        if ((rc = get_arr(pw, &hp, "pad_width")) < 0) return rc;
        if (!is_intdt(hp.a.dtype) && tsr_shape_size(hp.a.ndim, hp.a.shape) > 0) { fn_set_error("`pad_width` must be of integral type."); return TSR_ETYPE; }
        if (tsr_shape_size(hp.a.ndim, hp.a.shape) == 0) { fn_set_error("`pad_width` must be of integral type."); return TSR_ETYPE; }
    }
    int64_t pwv[TSR_MAXDIM][2];
    if ((rc = as_index_pairs(pw, nd, pwv, "pad_width")) < 0) return rc;
    const tsr_arg *ma = argk(args, nargs, 2);
    const char *mode = ma->kind == 2 && ma->str ? ma->str : "constant";
    static const char *const MODES[] = {"constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect", "symmetric", "wrap", "empty"};
    int m = -1;
    for (int i = 0; i < 11; i++) if (strcmp(mode, MODES[i]) == 0) m = i;
    if (m < 0) { fn_set_error("mode '%s' is not supported", mode); return TSR_EARG; }
    /* keyword arguments the mode does not take */
    static const char *const KW[] = {"constant_values", "end_values", "stat_length", "reflect_type"};
    const int allowed = m == 0 ? 0 : m == 2 ? 1 : (m >= 3 && m <= 6) ? 2 : (m == 7 || m == 8) ? 3 : -1;
    for (int k = 0; k < 4; k++)
        if (!is_none(argk(args, nargs, 3 + k)) && k != allowed) { fn_set_error("unsupported keyword arguments for mode '%s': {'%s'}", mode, KW[k]); return TSR_EARG; }
    /* _pad_simple */
    int64_t sh[TSR_MAXDIM];
    for (int d = 0; d < nd; d++) {
        if (pwv[d][0] > INT64_MAX / 4 || pwv[d][1] > INT64_MAX / 4) { fn_set_error("array is too big"); return TSR_EARG; }
        sh[d] = pwv[d][0] + arr->shape[d] + pwv[d][1];
    }
    if (tsr_shape_size(nd, sh) < 0) { fn_set_error("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."); return TSR_EARG; }
    own P;
    if ((rc = own_new(&P, dt, nd, sh)) < 0) return rc;
    tsr_array orig = P.a;
    for (int d = 0; d < nd; d++) v_slice(&orig, d, pwv[d][0], pwv[d][0] + arr->shape[d]);
    if ((rc = assign_bc(&orig, arr)) < 0) { own_free(&P); return rc; }
    const int64_t asize = tsr_shape_size(nd, arr->shape);
    if (m == 10) return own_emit(&res[0], &P);
    if (m != 0 && asize == 0) {
        for (int d = 0; d < nd; d++)
            if (arr->shape[d] == 0 && (pwv[d][0] || pwv[d][1])) {
                own_free(&P);
                fn_set_error("can't extend empty axis %d using modes other than 'constant' or 'empty'", d);
                return TSR_EARG;
            }
        return own_emit(&res[0], &P);
    }
    for (int ax = 0; ax < nd && rc == 0; ax++) {
        /* roi: the padded array restricted to the original area on the axes after ax */
        tsr_array roi = P.a;
        for (int d = ax + 1; d < nd; d++) v_slice(&roi, d, pwv[d][0], pwv[d][0] + arr->shape[d]);
        const int64_t L = pwv[ax][0], R = pwv[ax][1], n = roi.shape[ax];
        if (m == 0 || m == 2) {                          /* constant / linear_ramp: value pairs */
            const tsr_arg *va = argk(args, nargs, m == 0 ? 3 : 4);
            harr hv;
            if (is_none(va)) { tsr_array z = scalar_i64(&hv, 0); hv.a = z; }
            else if ((rc = get_arr(va, &hv, m == 0 ? "constant_values" : "end_values")) < 0) break;
            tsr_array pv;
            if ((rc = as_pairs(&hv.a, nd, &pv)) < 0) break;
            for (int s = 0; s < 2 && rc == 0; s++) {
                tsr_array val = pv;
                val.ndim = 0;
                val.offset += ax * pv.strides[0] + s * pv.strides[1];
                const int64_t w = s == 0 ? L : R;
                tsr_array area = s == 0 ? slice_at(&roi, ax, 0, L) : slice_at(&roi, ax, n - R, n);
                if (m == 0) {
                    if (is_intlike(dt) && is_floatdt(val.dtype) && rd_f64(elem0(&val), val.dtype) != rd_f64(elem0(&val), val.dtype) && w > 0 && tsr_shape_size(area.ndim, area.shape) > 0) {
                        fn_set_error("cannot convert float NaN to integer");
                        rc = TSR_EARG;
                        break;
                    }
                    rc = assign_bc(&area, &val);
                    continue;
                }
                /* linear_ramp: linspace(end_value, edge, num=w, endpoint=False, dtype=padded.dtype) along ax,
                   the right ramp reversed; any zero step on a side switches that side to (i / w) * delta */
                tsr_array edge = s == 0 ? slice_at(&roi, ax, L, L + 1) : slice_at(&roi, ax, n - R - 1, n - R);
                const double ev = rd_f64(elem0(&val), val.dtype);
                const int64_t lanes = tsr_shape_size(nd, edge.shape);
                int any_zero = 0;
                {
                    int64_t ix[TSR_MAXDIM] = {0};
                    for (int64_t q = 0; q < lanes && w > 0; q++) {
                        int64_t o = 0;
                        for (int d = 0; d < nd; d++) o += ix[d] * edge.strides[d];
                        const double delta = rd_f64(elem0(&edge) + o, dt) - ev;
                        if (delta / (double)w == 0) any_zero = 1;
                        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < edge.shape[d]) break; ix[d] = 0; }
                    }
                }
                int64_t ix[TSR_MAXDIM] = {0};
                for (int64_t q = 0; q < lanes; q++) {
                    int64_t oe = 0, oa = 0;
                    for (int d = 0; d < nd; d++) { oe += ix[d] * edge.strides[d]; oa += ix[d] * area.strides[d]; }
                    const double stop = rd_f64(elem0(&edge) + oe, dt), delta = stop - ev;
                    for (int64_t i = 0; i < w; i++) {
                        double y = (double)i;
                        if (any_zero) { y = y / (double)w; y = y * delta; }
                        else y = y * (delta / (double)w);
                        y = y + ev;
                        if (is_intdt(dt)) y = floor(y);           /* linspace(dtype=int): floor; bool is not an integer dtype */
                        const int64_t pos = s == 0 ? i : w - 1 - i;
                        tsr_array one = area;
                        one.ndim = 0;
                        one.offset += oa + pos * area.strides[ax];
                        harr hy;
                        tsr_array yv = scalar_f64(&hy, y);
                        if ((rc = assign_bc(&one, &yv)) < 0) break;
                    }
                    if (rc < 0) break;
                    for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < edge.shape[d]) break; ix[d] = 0; }
                }
            }
        } else if (m == 1) {                             /* edge */
            tsr_array le = slice_at(&roi, ax, L, L + 1), re = slice_at(&roi, ax, n - R - 1, n - R);
            tsr_array la = slice_at(&roi, ax, 0, L), ra = slice_at(&roi, ax, n - R, n);
            if ((rc = assign_bc(&la, &le)) == 0) rc = assign_bc(&ra, &re);
        } else if (m >= 3 && m <= 6) {                   /* maximum, mean, median, minimum */
            const int sm = m == 3 ? 0 : m == 6 ? 1 : m == 4 ? 2 : 3;
            int64_t sl[TSR_MAXDIM][2];
            int have = !is_none(argk(args, nargs, 5));
            if (have && (rc = as_index_pairs(argk(args, nargs, 5), nd, sl, "stat_length")) < 0) break;
            const int64_t li = L, ri = n - R, maxlen = ri - li;
            int64_t ll = have ? sl[ax][0] : maxlen, rl = have ? sl[ax][1] : maxlen;
            if (maxlen < ll) ll = maxlen;
            if (maxlen < rl) rl = maxlen;
            if ((ll == 0 || rl == 0) && (sm == 0 || sm == 1)) { fn_set_error("stat_length of 0 yields no value for padding"); rc = TSR_EARG; break; }
            own ls, rs;
            tsr_array lc = slice_at(&roi, ax, li, li + ll);
            if ((rc = pad_stat(&lc, ax, sm, dt, &ls)) < 0) break;
            if (ll == rl && rl == maxlen) rs = ls, rs.bytes = 0;
            else {
                tsr_array rcn = slice_at(&roi, ax, ri - rl, ri);
                if ((rc = pad_stat(&rcn, ax, sm, dt, &rs)) < 0) { own_free(&ls); break; }
            }
            tsr_array la = slice_at(&roi, ax, 0, L), ra = slice_at(&roi, ax, n - R, n);
            if ((rc = assign_bc(&la, &ls.a)) == 0) rc = assign_bc(&ra, &rs.a);
            if (rs.bytes) own_free(&rs);
            own_free(&ls);
        } else if (m == 7 || m == 8) {                   /* reflect / symmetric */
            const tsr_arg *rt = argk(args, nargs, 6);
            const int odd = rt->kind == 2 && rt->str && strcmp(rt->str, "odd") == 0;
            const int inc = m == 8;
            if (arr->shape[ax] == 1 && (L > 0 || R > 0)) {
                const int64_t np_ = P.a.shape[ax];
                tsr_array le = slice_at(&P.a, ax, L, L + 1), re = slice_at(&P.a, ax, np_ - R - 1, np_ - R);
                tsr_array la = slice_at(&P.a, ax, 0, L), ra = slice_at(&P.a, ax, np_ - R, np_);
                if ((rc = assign_bc(&la, &le)) == 0) rc = assign_bc(&ra, &re);
                continue;
            }
            int64_t left = L, right = R;
            const int64_t period = arr->shape[ax];
            while ((left > 0 || right > 0) && rc == 0) {
                int64_t old = roi.shape[ax] - right - left, eoff;
                if (inc) { old = old / period * period; eoff = 1; }
                else { old = (old - 1) / (period - 1) * (period - 1) + 1; eoff = 0; old -= 1; }
                if (left > 0) {
                    const int64_t cl = old < left ? old : left;
                    const int64_t stop = left - eoff, start = stop + cl;           /* slice(start, stop, -1) */
                    tsr_array chunk = roi;
                    /* elements start, start-1, ..., stop+1 */
                    chunk.offset += start * roi.strides[ax];
                    chunk.shape[ax] = cl;
                    chunk.strides[ax] = -roi.strides[ax];
                    tsr_array area = slice_at(&roi, ax, left - cl, left);
                    if (odd) {
                        own tmp, two_e;
                        tsr_array ed = slice_at(&roi, ax, left, left + 1);
                        harr h2;
                        tsr_array two = scalar_i64(&h2, 2);
                        if ((rc = bop(OP_MUL, &two, 'i', &ed, 0, &two_e)) < 0) break;
                        rc = bop(OP_SUB, &two_e.a, 0, &chunk, 0, &tmp);
                        own_free(&two_e);
                        if (rc < 0) break;
                        rc = assign_bc(&area, &tmp.a);
                        own_free(&tmp);
                    } else rc = assign_bc(&area, &chunk);
                    left -= cl;
                }
                if (right > 0 && rc == 0) {
                    const int64_t cl = old < right ? old : right;
                    const int64_t nn = roi.shape[ax];
                    const int64_t start = nn + (-right + eoff - 2);                  /* python negative index */
                    tsr_array chunk = roi;
                    chunk.offset += start * roi.strides[ax];
                    chunk.shape[ax] = cl;
                    chunk.strides[ax] = -roi.strides[ax];
                    const int64_t ast = nn - right;
                    tsr_array area = slice_at(&roi, ax, ast, ast + cl);
                    if (odd) {
                        own tmp, two_e;
                        tsr_array ed = slice_at(&roi, ax, nn - right - 1, nn - right);
                        harr h2;
                        tsr_array two = scalar_i64(&h2, 2);
                        if ((rc = bop(OP_MUL, &two, 'i', &ed, 0, &two_e)) < 0) break;
                        rc = bop(OP_SUB, &two_e.a, 0, &chunk, 0, &tmp);
                        own_free(&two_e);
                        if (rc < 0) break;
                        rc = assign_bc(&area, &tmp.a);
                        own_free(&tmp);
                    } else rc = assign_bc(&area, &chunk);
                    right -= cl;
                }
            }
        } else if (m == 9) {                             /* wrap */
            int64_t left = L, right = R;
            const int64_t operiod = roi.shape[ax] - right - left;
            while ((left > 0 || right > 0) && rc == 0) {
                int64_t period = roi.shape[ax] - right - left;
                period = period / operiod * operiod;
                int64_t nl = 0, nr = 0;
                const int64_t nn = roi.shape[ax];
                if (left > 0) {
                    const int64_t se = left + period, ss = se - (period < left ? period : left);
                    tsr_array chunk = slice_at(&roi, ax, ss, se);
                    tsr_array area;
                    if (left > period) { area = slice_at(&roi, ax, left - period, left); nl = left - period; }
                    else area = slice_at(&roi, ax, 0, left);
                    rc = assign_bc(&area, &chunk);
                }
                if (right > 0 && rc == 0) {
                    const int64_t ss = nn - right - period, se = ss + (period < right ? period : right);
                    tsr_array chunk = slice_at(&roi, ax, ss, se);
                    tsr_array area;
                    if (right > period) { area = slice_at(&roi, ax, nn - right, nn - right + period); nr = right - period; }
                    else area = slice_at(&roi, ax, nn - right, nn);
                    rc = assign_bc(&area, &chunk);
                }
                left = nl;
                right = nr;
            }
        }
    }
    if (rc < 0) { own_free(&P); return rc; }
    return own_emit(&res[0], &P);
}

/* ================================================================ writers (the first argument is modified in place) */

/* the flat (C-order) element pointers of a view: element i of `a` */
static char *flat_ptr(const tsr_array *a, int64_t i)
{
    int64_t off = 0;
    for (int d = a->ndim - 1; d >= 0; d--) {
        const int64_t n = a->shape[d];
        off += (i % n) * a->strides[d];
        i /= n;
    }
    return elem0(a) + off;
}

/* one value into one element: unsafe cast (setitem), as numpy's put/putmask/place/fill_diagonal/put_along_axis */
static void put_elem(char *dst, int ddt, const char *src, int sdt)
{
    const int64_t one = 1, z = 0;
    tsr_copy(sdt, ddt, 1, &one, src, &z, dst, &z);
}

/* flattened values of an argument (a view over an owned copy) */
static int values_flat(const tsr_arg *v, own *o, const char *what)
{
    harr h;
    int rc = get_arr(v, &h, what);
    if (rc < 0) return rc;
    return own_ravel(o, &h.a);
}

/* NumPy converts Python values (a number, a list) for an integer destination with int() and a range check: NaN is
   a ValueError, infinity and values outside the destination's range an OverflowError, other floats truncate as the
   unsafe copy does. An NDArray of values is cast unchecked (setitem's unsafe cast). */
static int py_values_check(const tsr_arg *va, const tsr_array *vals, int ddt)
{
    if (!(va->kind == 1 || (va->kind == 3 && (va->flags & 2)))) return TSR_OK;
    if (ddt != TSR_I64 && ddt != TSR_I32 && ddt != TSR_U8) return TSR_OK;
    const int64_t n = tsr_shape_size(vals->ndim, vals->shape);
    for (int64_t i = 0; i < n; i++) {
        const char *p = flat_ptr(vals, i);
        int64_t x;
        if (is_floatdt(vals->dtype)) {
            const double v = rd_f64(p, vals->dtype);
            if (v != v) { fn_set_error("cannot convert float NaN to integer"); return TSR_EARG; }
            if (isinf(v)) { fn_set_error("cannot convert float infinity to integer"); return TSR_EARG; }
            if (!(v >= -9223372036854775808.0 && v < 9223372036854775808.0)) { fn_set_error("Python int too large to convert to C long"); return TSR_EARG; }
            x = (int64_t)v;
        } else if (vals->dtype == TSR_C128) return TSR_OK;
        else x = rd_i64(p, vals->dtype);
        if ((ddt == TSR_I32 && (x < INT32_MIN || x > INT32_MAX)) || (ddt == TSR_U8 && (x < 0 || x > 255))) {
            fn_set_error("Python integer %lld out of bounds for %s", (long long)x, dtname(ddt));
            return TSR_EARG;
        }
    }
    return TSR_OK;
}

/* put(a, ind, v, mode='raise') */
static int r_put(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const tsr_array *a = &args[0].arr;
    const tsr_arg *ma = argk(args, nargs, 3);
    int mode = 0;
    if (ma->kind == 2 && ma->str) {
        if (strcmp(ma->str, "raise") == 0) mode = 0;
        else if (strcmp(ma->str, "wrap") == 0) mode = 1;
        else if (strcmp(ma->str, "clip") == 0) mode = 2;
        else { fn_set_error("clipmode must be one of 'clip', 'raise', or 'wrap' (got '%s')", ma->str); return TSR_EARG; }
    }
    own iv, vv;
    harr hi;
    int rc = get_arr(argk(args, nargs, 1), &hi, "ind");
    if (rc < 0) return rc;
    const tsr_arg *ia = argk(args, nargs, 1);
    if (is_floatdt(hi.a.dtype) && ia->kind == 3 && !(ia->flags & 2) && tsr_shape_size(hi.a.ndim, hi.a.shape) > 0) {
        fn_set_error("Cannot cast array data from dtype('%s') to dtype('int64') according to the rule 'safe'", dtname(hi.a.dtype));
        return TSR_EARG;
    }
    if ((rc = own_ravel(&iv, &hi.a)) < 0) return rc;
    if ((rc = values_flat(argk(args, nargs, 2), &vv, "v")) < 0) { own_free(&iv); return rc; }
    if ((rc = py_values_check(argk(args, nargs, 2), &vv.a, a->dtype)) < 0) { own_free(&iv); own_free(&vv); return rc; }
    const int64_t N = tsr_shape_size(a->ndim, a->shape), ni = iv.a.shape[0], nv = vv.a.shape[0];
    if (nv == 0 || ni == 0) { own_free(&iv); own_free(&vv); return TSR_OK; }
    if (N == 0) { own_free(&iv); own_free(&vv); fn_set_error("cannot replace elements of an empty array"); return TSR_EARG; }
    for (int64_t k = 0; k < ni && rc == 0; k++) {
        const char *pi = elem0(&iv.a) + k * iv.a.strides[0];
        int64_t j;
        if (is_floatdt(iv.a.dtype)) {
            const double x = rd_f64(pi, iv.a.dtype);
            if (!(x > -9.2e18 && x < 9.2e18)) { fn_set_error("cannot convert float NaN to integer"); rc = TSR_EARG; break; }
            j = (int64_t)x;
        } else j = rd_i64(pi, iv.a.dtype);
        if (mode == 0) {
            if (j < -N || j >= N) { fn_set_error("index %lld is out of bounds for axis 0 with size %lld", (long long)j, (long long)N); rc = TSR_EARG; break; }
            if (j < 0) j += N;
        } else if (mode == 1) { j %= N; if (j < 0) j += N; }
        else j = j < 0 ? 0 : (j >= N ? N - 1 : j);
        put_elem(flat_ptr(a, j), a->dtype, elem0(&vv.a) + (k % nv) * vv.a.strides[0], vv.a.dtype);
    }
    own_free(&iv);
    own_free(&vv);
    return rc;
}

/* putmask(a, mask, values) / place(arr, mask, vals) */
static int r_putmask(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int place = *(const int *)ctx;
    const tsr_array *a = &args[0].arr;
    own mv, vv;
    int rc = values_flat(argk(args, nargs, 1), &mv, "mask");
    if (rc < 0) return rc;
    if ((rc = values_flat(argk(args, nargs, 2), &vv, place ? "vals" : "values")) < 0) { own_free(&mv); return rc; }
    if ((rc = py_values_check(argk(args, nargs, 2), &vv.a, a->dtype)) < 0) { own_free(&mv); own_free(&vv); return rc; }
    const int64_t N = tsr_shape_size(a->ndim, a->shape), nm = mv.a.shape[0], nv = vv.a.shape[0];
    /* values given as an NDArray are converted with 'safe' casting; Python scalars and lists are forced */
    const tsr_arg *va = argk(args, nargs, 2);
    if (va->kind == 3 && !(va->flags & 2) && vv.a.dtype != a->dtype) {
        static const char safe[7][8] = {"1000001", "1100001", "1010001", "1011001", "1111101", "1111111", "0000001"};
        if (safe[vv.a.dtype][a->dtype] != '1') {
            own_free(&mv); own_free(&vv);
            fn_set_error("Cannot cast array data from dtype('%s') to dtype('%s') according to the rule 'safe'", dtname(vv.a.dtype), dtname(a->dtype));
            return TSR_ETYPE;
        }
    }
    if (nm != N) {
        own_free(&mv); own_free(&vv);
        fn_set_error("%s: mask and data must be the same size", place ? "place" : "putmask");
        return TSR_EARG;
    }
    int64_t cnt = 0, hit = 0;
    for (int64_t i = 0; i < N; i++) if (rd_i64(elem0(&mv.a) + i * mv.a.strides[0], mv.a.dtype) != 0 || (is_floatdt(mv.a.dtype) && rd_f64(elem0(&mv.a) + i * mv.a.strides[0], mv.a.dtype) != 0)) hit++;
    if (nv == 0) {
        own_free(&mv); own_free(&vv);
        if (place && hit > 0) { fn_set_error("Cannot insert from an empty array!"); return TSR_EARG; }
        return TSR_OK;
    }
    for (int64_t i = 0; i < N; i++) {
        const char *pm = elem0(&mv.a) + i * mv.a.strides[0];
        const int on = is_floatdt(mv.a.dtype) ? rd_f64(pm, mv.a.dtype) != 0 : rd_i64(pm, mv.a.dtype) != 0;
        if (!on) continue;
        const int64_t k = place ? cnt++ % nv : i % nv;
        put_elem(flat_ptr(a, i), a->dtype, elem0(&vv.a) + k * vv.a.strides[0], vv.a.dtype);
    }
    own_free(&mv);
    own_free(&vv);
    return TSR_OK;
}
static const int PUTMASK = 0, PLACE = 1;

/* copyto(dst, src, casting='same_kind', where=True) */
static int r_copyto(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const tsr_array *dst = &args[0].arr;
    harr hs, hw;
    const tsr_arg *sa = argk(args, nargs, 1);
    int rc = get_arr(sa, &hs, "src");
    if (rc < 0) return rc;
    const tsr_arg *ca = argk(args, nargs, 2);
    int casting = 3;
    if (ca->kind == 2 && ca->str) {
        static const char *const C[] = {"no", "equiv", "safe", "same_kind", "unsafe"};
        casting = -1;
        for (int i = 0; i < 5; i++) if (strcmp(ca->str, C[i]) == 0) casting = i;
        if (casting < 0) { fn_set_error("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'"); return TSR_EARG; }
    }
    /* can_cast(src, dst.dtype, casting); a Python scalar is weak */
    const int sd = hs.a.dtype, dd = dst->dtype;
    const char wk = weak_kind(sa);
    int ok;
    if (casting == 4 || sd == dd) ok = 1;
    else if (casting <= 1) ok = 0;
    else if (wk == 'b') ok = 1;
    else if (wk == 'i') ok = dd != TSR_BOOL;
    else if (wk == 'f') ok = is_floatdt(dd) || dd == TSR_C128;
    else {
        static const char safe[7][8] = {"1000001", "1100001", "1010001", "1011001", "1111101", "1111111", "0000001"};
        static const char same_kind[7][8] = {"1100001", "1100001", "1111001", "1111001", "1111101", "1111111", "0000001"};
        ok = (casting == 2 ? safe : same_kind)[sd][dd] == '1';
    }
    if (!ok) {
        static const char *const C[] = {"no", "equiv", "safe", "same_kind", "unsafe"};
        fn_set_error("Cannot cast %s from dtype('%s') to dtype('%s') according to the rule '%s'", wk ? "scalar" : "array data", dtname(sd), dtname(dd), C[casting]);
        return TSR_ETYPE;
    }
    tsr_array s = hs.a;
    if (s.ndim > dst->ndim || v_broadcast(&s, dst->ndim, dst->shape) < 0) {
        fn_set_error("could not broadcast input array into the destination shape");
        return TSR_EARG;
    }
    const tsr_arg *wa = argk(args, nargs, 3);
    if (is_none(wa) || (wa->kind == 4 && wa->num != 0)) return assign_bc(dst, &s);
    if (wa->kind == 4) return TSR_OK;                    /* where=False */
    if ((rc = get_arr(wa, &hw, "where")) < 0) return rc;
    if (hw.a.dtype != TSR_BOOL) { fn_set_error("Cannot cast array data from dtype('%s') to dtype('bool') according to the rule 'safe'", dtname(hw.a.dtype)); return TSR_ETYPE; }
    tsr_array w = hw.a;
    if (w.ndim > dst->ndim || v_broadcast(&w, dst->ndim, dst->shape) < 0) { fn_set_error("could not broadcast where mask into the destination shape"); return TSR_EARG; }
    const int64_t n = tsr_shape_size(dst->ndim, dst->shape);
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t od = 0, os = 0, ow = 0;
        for (int d = 0; d < dst->ndim; d++) { od += ix[d] * dst->strides[d]; os += ix[d] * s.strides[d]; ow += ix[d] * w.strides[d]; }
        if (*(elem0(&w) + ow)) put_elem(elem0(dst) + od, dd, elem0(&s) + os, sd);
        for (int d = dst->ndim - 1; d >= 0; d--) { if (++ix[d] < dst->shape[d]) break; ix[d] = 0; }
    }
    return TSR_OK;
}

/* fill_diagonal(a, val, wrap=False) */
static int r_fill_diagonal(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const tsr_array *a = &args[0].arr;
    if (a->ndim < 2) { fn_set_error("array must be at least 2-d"); return TSR_EARG; }
    const tsr_arg *wa = argk(args, nargs, 2);
    const int wrap = !is_none(wa) && wa->num != 0;
    int64_t step, end = tsr_shape_size(a->ndim, a->shape);
    if (a->ndim == 2) {
        step = a->shape[1] + 1;
        if (!wrap && a->shape[1] * a->shape[1] < end) end = a->shape[1] * a->shape[1];
    } else {
        for (int d = 1; d < a->ndim; d++) if (a->shape[d] != a->shape[0]) { fn_set_error("All dimensions of input must be of equal length"); return TSR_EARG; }
        step = 1;
        int64_t c = 1;
        for (int d = 0; d < a->ndim - 1; d++) { c *= a->shape[d]; step += c; }
    }
    own vv;
    int rc = values_flat(argk(args, nargs, 1), &vv, "val");
    if (rc < 0) return rc;
    if ((rc = py_values_check(argk(args, nargs, 1), &vv.a, a->dtype)) < 0) { own_free(&vv); return rc; }
    const int64_t nv = vv.a.shape[0];
    if (nv > 0)
        for (int64_t i = 0, k = 0; i < end; i += step, k++)
            put_elem(flat_ptr(a, i), a->dtype, elem0(&vv.a) + (k % nv) * vv.a.strides[0], vv.a.dtype);
    own_free(&vv);
    return TSR_OK;
}

/* put_along_axis(arr, indices, values, axis) */
static int r_put_along_axis(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const tsr_array *arr = &args[0].arr;
    harr hi, hv;
    int rc = get_arr(argk(args, nargs, 1), &hi, "indices");
    if (rc == 0) rc = get_arr(argk(args, nargs, 2), &hv, "values");
    if (rc < 0) return rc;
    if ((rc = py_values_check(argk(args, nargs, 2), &hv.a, arr->dtype)) < 0) return rc;
    const tsr_arg *aa = argk(args, nargs, 3);
    tsr_array a = *arr;
    int ax;
    if (is_none(aa)) {
        /* numpy: arr = np.array(arr.flat), which shares memory with a C-contiguous arr and is read-only otherwise */
        if (hi.a.ndim != 1) { fn_set_error("when axis=None, `indices` must have a single dimension."); return TSR_EARG; }
        if (!tsr_array_is_contiguous(arr)) { fn_set_error("assignment destination is read-only"); return TSR_EARG; }
        a.ndim = 1;
        a.shape[0] = tsr_shape_size(arr->ndim, arr->shape);
        a.strides[0] = tsr_itemsize(arr->dtype);
        ax = 0;
    } else {
        int64_t a64;
        if ((rc = int_param(aa, "axis", &a64)) < 0 || (rc = norm_axis(a64, a.ndim, &ax)) < 0) return rc;
    }
    if (!is_intdt(hi.a.dtype)) { fn_set_error("`indices` must be an integer array"); return TSR_EARG; }
    if (hi.a.ndim != a.ndim) { fn_set_error("`indices` and `arr` must have the same number of dimensions"); return TSR_EARG; }
    const int nd = a.ndim;
    int64_t sh[TSR_MAXDIM];
    for (int d = 0; d < nd; d++) {
        if (d == ax) { sh[d] = hi.a.shape[d]; continue; }
        const int64_t x = a.shape[d], y = hi.a.shape[d];
        if (x != y && x != 1 && y != 1) { fn_set_error("shape mismatch: indexing arrays could not be broadcast together"); return TSR_EARG; }
        sh[d] = y == 1 ? x : y;
        if (x == 1 && y != 1) sh[d] = y;
    }
    tsr_array v = hv.a;
    while (v.ndim > nd && v.shape[0] == 1) {
        memmove(v.shape, v.shape + 1, sizeof(int64_t) * (size_t)(v.ndim - 1));
        memmove(v.strides, v.strides + 1, sizeof(int64_t) * (size_t)(v.ndim - 1));
        v.ndim--;
    }
    if (v.ndim > nd || v_broadcast(&v, nd, sh) < 0) { fn_set_error("shape mismatch: value array could not be broadcast to indexing result"); return TSR_EARG; }
    const int64_t n = tsr_shape_size(nd, sh), N = a.shape[ax];
    /* validate every index first (numpy raises before writing) */
    int64_t ix[TSR_MAXDIM] = {0};
    for (int pass = 0; pass < 2; pass++) {
        memset(ix, 0, sizeof ix);
        for (int64_t q = 0; q < n; q++) {
            int64_t oi = 0, oa = 0, ov = 0;
            for (int d = 0; d < nd; d++) {
                oi += (hi.a.shape[d] == 1 ? 0 : ix[d]) * hi.a.strides[d];
                ov += ix[d] * v.strides[d];
                if (d != ax) oa += (a.shape[d] == 1 ? 0 : ix[d]) * a.strides[d];
            }
            int64_t j = rd_i64(elem0(&hi.a) + oi, hi.a.dtype);
            if (j < -N || j >= N) {
                fn_set_error("index %lld is out of bounds for axis %d with size %lld", (long long)j, ax, (long long)N);
                return TSR_EARG;
            }
            if (j < 0) j += N;
            if (pass == 1) put_elem(elem0(&a) + oa + j * a.strides[ax], a.dtype, elem0(&v) + ov, v.dtype);
            for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < sh[d]) break; ix[d] = 0; }
        }
    }
    return TSR_OK;
}

/* ================================================================ round, isclose, allclose */

/* round(a, decimals=0): numpy.round / around. NumPy's PyArray_Round: for decimals >= 0, multiply by
   10**decimals, round to nearest even (rint), divide; for decimals < 0, divide, rint, multiply. The result
   keeps the input dtype; integer (and bool) input with decimals >= 0 is returned unchanged. */
static int r_round(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    int64_t dec = 0;
    if (!is_none(argk(args, nargs, 1)) && (rc = int_param(argk(args, nargs, 1), "decimals", &dec)) < 0) return rc;
    const int dt = h.a.dtype;
    own z;
    if ((rc = own_from(&z, &h.a, -1)) < 0) return rc;      /* contiguous copy, same dtype */
    const int64_t n = tsr_shape_size(z.a.ndim, z.a.shape);
    if (is_intlike(dt) && dec >= 0) return own_emit(&res[0], &z);   /* numpy: unchanged */
    const int ad = dec < 0 ? (int)-dec : (int)dec;
    const double f = pow(10.0, (double)ad);
    if (dt == TSR_F64 || dt == TSR_C128) {
        double *p = (double *)z.a.data;
        const int64_t m = dt == TSR_C128 ? 2 * n : n;
        for (int64_t i = 0; i < m; i++)
            p[i] = dec >= 0 ? rint(p[i] * f) / f : rint(p[i] / f) * f;
    } else if (dt == TSR_F32) {
        float *p = (float *)z.a.data;
        const float ff = (float)f;
        for (int64_t i = 0; i < n; i++)
            p[i] = dec >= 0 ? rintf(p[i] * ff) / ff : rintf(p[i] / ff) * ff;
    } else if (dt == TSR_I64) {
        int64_t *p = (int64_t *)z.a.data;                  /* decimals < 0 */
        for (int64_t i = 0; i < n; i++) p[i] = (int64_t)(rint((double)p[i] / f) * f);
    } else if (dt == TSR_I32) {
        int32_t *p = (int32_t *)z.a.data;
        for (int64_t i = 0; i < n; i++) p[i] = (int32_t)(rint((double)p[i] / f) * f);
    } else if (dt == TSR_U8) {
        uint8_t *p = (uint8_t *)z.a.data;
        for (int64_t i = 0; i < n; i++) { double r = rint((double)p[i] / f) * f; p[i] = (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r); }
    }
    return own_emit(&res[0], &z);
}

/* read one element as a complex value (re, im); real dtypes give im = 0 */
static void rd_c128(const char *p, int dt, double *re, double *im)
{
    if (dt == TSR_C128) { memcpy(re, p, 8); memcpy(im, p + 8, 8); }
    else { *re = rd_f64(p, dt); *im = 0.0; }
}

/* numpy.isclose element rule: |a-b| <= atol + rtol*|b| where both finite; a == b for the non-finite pair
   (inf == inf of the same sign is close, inf vs finite is not, NaN is not unless equal_nan and both are NaN). */
static int isclose_elem(double are, double aim, double bre, double bim, double rtol, double atol, int equal_nan)
{
    const int fa = isfinite(are) && isfinite(aim), fb = isfinite(bre) && isfinite(bim);
    if (fa && fb) return hypot(are - bre, aim - bim) <= atol + rtol * hypot(bre, bim);
    if (are == bre && aim == bim) return 1;
    if (equal_nan && (isnan(are) || isnan(aim)) && (isnan(bre) || isnan(bim))) return 1;
    return 0;
}

/* shared body of isclose (per-element bool array) and allclose (scalar bool). all != NULL: allclose. */
static int isclose_body(const tsr_arg *args, int nargs, tsr_result *res, int *all_out)
{
    harr ha, hb;
    int rc = get_arr(argk(args, nargs, 0), &ha, "a");
    if (rc < 0) return rc;
    if ((rc = get_arr(argk(args, nargs, 1), &hb, "b")) < 0) return rc;
    double rtol = 1e-05, atol = 1e-08;
    int64_t eqnan = 0;
    if (!is_none(argk(args, nargs, 2)) && (rc = num_param(argk(args, nargs, 2), "rtol", &rtol)) < 0) return rc;
    if (!is_none(argk(args, nargs, 3)) && (rc = num_param(argk(args, nargs, 3), "atol", &atol)) < 0) return rc;
    if (!is_none(argk(args, nargs, 4)) && (rc = int_param(argk(args, nargs, 4), "equal_nan", &eqnan)) < 0) return rc;
    int32_t nd;
    int64_t sh[TSR_MAXDIM];
    if ((rc = bshape(ha.a.ndim, ha.a.shape, hb.a.ndim, hb.a.shape, &nd, sh)) < 0) return rc;
    tsr_array va = ha.a, vb = hb.a;
    if ((rc = v_broadcast(&va, nd, sh)) < 0) return rc;
    if ((rc = v_broadcast(&vb, nd, sh)) < 0) return rc;
    const int64_t n = tsr_shape_size(nd, sh);
    own z;
    if (!all_out) { if ((rc = own_new(&z, TSR_BOOL, nd, sh)) < 0) return rc; }
    int all = 1;
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t oa = 0, ob = 0;
        for (int d = 0; d < nd; d++) { oa += ix[d] * va.strides[d]; ob += ix[d] * vb.strides[d]; }
        double are, aim, bre, bim;
        rd_c128(elem0(&va) + oa, va.dtype, &are, &aim);
        rd_c128(elem0(&vb) + ob, vb.dtype, &bre, &bim);
        const int c = isclose_elem(are, aim, bre, bim, rtol, atol, (int)eqnan);
        if (all_out) { if (!c) { all = 0; break; } }
        else ((uint8_t *)z.a.data)[q] = (uint8_t)c;
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < sh[d]) break; ix[d] = 0; }
    }
    if (all_out) { *all_out = all; return TSR_OK; }
    return own_emit(&res[0], &z);
}

static int r_isclose(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    return isclose_body(args, nargs, res, NULL);
}

static int r_allclose(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int all = 1;
    int rc = isclose_body(args, nargs, res, &all);
    if (rc < 0) return rc;
    memset(&res[0], 0, sizeof res[0]);
    res[0].kind = 4;
    res[0].num = all ? 1.0 : 0.0;
    return TSR_OK;
}

/* ================================================================ result_type / promote_types */

static int dtype_from_name(const char *s, int *dt)
{
    static const struct { const char *n; int d; } T[] = {
        {"float64", TSR_F64}, {"float", TSR_F64}, {"f8", TSR_F64}, {"double", TSR_F64}, {"d", TSR_F64}, {"float_", TSR_F64},
        {"float32", TSR_F32}, {"f4", TSR_F32}, {"f", TSR_F32}, {"single", TSR_F32},
        {"int64", TSR_I64}, {"int", TSR_I64}, {"i8", TSR_I64}, {"intp", TSR_I64}, {"long", TSR_I64}, {"int_", TSR_I64},
        {"int32", TSR_I32}, {"i4", TSR_I32}, {"intc", TSR_I32},
        {"uint8", TSR_U8}, {"u1", TSR_U8}, {"ubyte", TSR_U8}, {"B", TSR_U8},
        {"bool", TSR_BOOL}, {"?", TSR_BOOL}, {"bool_", TSR_BOOL},
        {"complex128", TSR_C128}, {"complex", TSR_C128}, {"c16", TSR_C128}, {"D", TSR_C128}, {"cdouble", TSR_C128},
    };
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++)
        if (strcmp(s, T[i].n) == 0) { *dt = T[i].d; return TSR_OK; }
    fn_set_error("data type '%s' not understood", s);
    return TSR_EARG;
}

/* the dtype of a result_type/promote_types argument: an array's dtype, a dtype name, or a number's strong dtype */
static int arg_to_dtype(const tsr_arg *a, int *dt)
{
    if (a->kind == 3) { *dt = a->arr.dtype; return TSR_OK; }
    if (a->kind == 2 && a->str) return dtype_from_name(a->str, dt);
    if (a->kind == 1) { *dt = (a->flags & 1) ? TSR_I64 : TSR_F64; return TSR_OK; }
    if (a->kind == 4) { *dt = TSR_BOOL; return TSR_OK; }
    fn_set_error("result_type: arguments must be arrays, dtypes or numbers");
    return TSR_EARG;
}

/* result_type(*arrays_and_dtypes): the dtype NumPy's promotion yields (numpy.result_type) */
static int r_result_type(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const tsr_arg *items = args;
    int64_t n = nargs;
    if (nargs == 1 && args[0].kind == 5) { items = args[0].items; n = args[0].count; }
    if (n <= 0) { fn_set_error("at least one array or dtype is required"); return TSR_EARG; }
    int dt = -1;
    for (int64_t i = 0; i < n; i++) {
        int d, rc = arg_to_dtype(&items[i], &d);
        if (rc < 0) return rc;
        dt = dt < 0 ? d : promote2(dt, d);
    }
    return fn_result_str(&res[0], dtname(dt));
}

/* promote_types(type1, type2): the dtype both can be safely cast to (numpy.promote_types) */
static int r_promote_types(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int d1, d2, rc;
    if ((rc = arg_to_dtype(argk(args, nargs, 0), &d1)) < 0) return rc;
    if ((rc = arg_to_dtype(argk(args, nargs, 1), &d2)) < 0) return rc;
    return fn_result_str(&res[0], dtname(promote2(d1, d2)));
}

/* einsum(subscripts, *operands): Einstein summation (numpy.einsum), real float64 operands.
   General but unoptimised: it accumulates over every distinct label with one odometer loop. Supports
   repeated labels (diagonal/trace), implicit and explicit output, and a single ellipsis (with broadcasting
   of the ellipsis dimensions). Complex operands are not supported here. */
#define EINSUM_MAXOP 16
#define EINSUM_MAXLBL 64

static int r_einsum(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 1 || args[0].kind != 2 || !args[0].str) {
        fn_set_error("einsum: a subscripts string is required"); return TSR_EARG;
    }
    const tsr_arg *ops; int nop;
    if (nargs == 2 && args[1].kind == 5) { ops = args[1].items; nop = (int)args[1].count; }
    else { ops = args + 1; nop = nargs - 1; }
    if (nop < 1 || nop > EINSUM_MAXOP) { fn_set_error("einsum: between 1 and %d operands are required", EINSUM_MAXOP); return TSR_EARG; }
    for (int o = 0; o < nop; o++) if (ops[o].kind != 3) { fn_set_error("einsum: operand %d must be an array", o + 1); return TSR_EARG; }

    /* strip spaces, split on "->" */
    char buf[256]; size_t bl = 0;
    for (const char *s = args[0].str; *s; s++) if (*s != ' ') { if (bl + 1 >= sizeof buf) { fn_set_error("einsum: subscripts too long"); return TSR_EARG; } buf[bl++] = *s; }
    buf[bl] = 0;
    char *arrow = strstr(buf, "->");
    const int explicit_out = arrow != NULL;
    char inpart[256], outpart[256];
    if (arrow) { size_t k = (size_t)(arrow - buf); memcpy(inpart, buf, k); inpart[k] = 0; snprintf(outpart, sizeof outpart, "%s", arrow + 2); }
    else { snprintf(inpart, sizeof inpart, "%s", buf); outpart[0] = 0; }

    /* split the input part into one term per operand */
    char *terms[EINSUM_MAXOP]; int nterms = 0;
    { char *p = inpart; terms[nterms++] = p; for (; *p; p++) if (*p == ',') { *p = 0; if (nterms >= EINSUM_MAXOP) { fn_set_error("einsum: too many terms"); return TSR_EARG; } terms[nterms++] = p + 1; } }
    if (nterms != nop) { fn_set_error("einsum: %d operands but %d subscript terms", nop, nterms); return TSR_EARG; }

    /* pass A: register named labels, find the ellipsis breadth E */
    int lblmap[256]; for (int i = 0; i < 256; i++) lblmap[i] = -1;
    char idchar[EINSUM_MAXLBL]; int nnamed = 0; int E = 0;
    for (int o = 0; o < nop; o++) {
        const char *t = terms[o]; int letters = 0, dots = 0;
        for (const char *p = t; *p; p++) {
            if (*p == '.') { dots++; }
            else if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) {
                letters++;
                if (lblmap[(unsigned char)*p] < 0) { if (nnamed >= EINSUM_MAXLBL) { fn_set_error("einsum: too many distinct labels"); return TSR_EARG; } idchar[nnamed] = *p; lblmap[(unsigned char)*p] = nnamed++; }
            } else { fn_set_error("einsum: invalid character '%c' in subscripts", *p); return TSR_EARG; }
        }
        if (dots != 0 && dots != 3) { fn_set_error("einsum: '.' may only appear in an ellipsis '...'"); return TSR_EARG; }
        const int32_t nd = ops[o].arr.ndim;
        if (dots == 3) { int cov = nd - letters; if (cov < 0) { fn_set_error("einsum: operand %d has fewer dimensions than its subscript", o + 1); return TSR_EARG; } if (cov > E) E = cov; }
        else if (letters != nd) { fn_set_error("einsum: operand %d has %d dimensions but %d labels", o + 1, nd, letters); return TSR_EARG; }
    }
    const int base_ell = nnamed;
    const int nlab = nnamed + E;
    if (nlab > EINSUM_MAXLBL) { fn_set_error("einsum: too many distinct labels"); return TSR_EARG; }
    (void)idchar;

    /* pass B: per operand, the label id and element stride of each axis; also label sizes (max, broadcast on 1) */
    int   olab[EINSUM_MAXOP][TSR_MAXDIM];
    int64_t ostr[EINSUM_MAXOP][TSR_MAXDIM];
    int64_t osz[EINSUM_MAXOP][TSR_MAXDIM];
    int64_t lsize[EINSUM_MAXLBL]; for (int i = 0; i < nlab; i++) lsize[i] = 0;
    for (int o = 0; o < nop; o++) {
        const int32_t nd = ops[o].arr.ndim; const int64_t *shp = ops[o].arr.shape;
        int letters = 0; for (const char *p = terms[o]; *p; p++) if (*p != '.') letters++;
        const int cov = (strstr(terms[o], "...") != NULL) ? nd - letters : 0;
        int ax = 0;
        for (const char *p = terms[o]; *p; ) {
            if (*p == '.') { for (int j = 0; j < cov; j++) { olab[o][ax] = base_ell + (E - cov + j); ax++; } p += 3; }
            else { olab[o][ax] = lblmap[(unsigned char)*p]; ax++; p++; }
        }
        /* C-order element strides for this operand's shape */
        int64_t st = 1;
        for (int d = nd - 1; d >= 0; d--) { ostr[o][d] = st; osz[o][d] = shp[d]; st *= shp[d]; }
        for (int d = 0; d < nd; d++) {
            const int L = olab[o][d]; const int64_t s = shp[d];
            if (lsize[L] == 0 || (lsize[L] == 1 && s != 1)) lsize[L] = s;
            if (s != 1 && lsize[L] != s) { fn_set_error("einsum: label size mismatch (%lld vs %lld)", (long long)lsize[L], (long long)s); return TSR_EARG; }
        }
    }

    /* output labels: explicit (parse outpart) or implicit (ellipsis dims, then once-only labels sorted) */
    int outlab[EINSUM_MAXLBL]; int nout = 0;
    if (explicit_out) {
        for (const char *p = outpart; *p; ) {
            if (*p == '.') { for (int j = 0; j < E; j++) outlab[nout++] = base_ell + j; p += 3; }
            else if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) {
                const int L = lblmap[(unsigned char)*p];
                if (L < 0) { fn_set_error("einsum: output label '%c' does not appear in the input", *p); return TSR_EARG; }
                outlab[nout++] = L; p++;
            } else { fn_set_error("einsum: invalid character '%c' in the output subscript", *p); return TSR_EARG; }
        }
    } else {
        for (int j = 0; j < E; j++) outlab[nout++] = base_ell + j;         /* ellipsis dims first */
        int occ[EINSUM_MAXLBL]; for (int i = 0; i < nnamed; i++) occ[i] = 0;
        for (int o = 0; o < nop; o++) { const int32_t nd = ops[o].arr.ndim; for (int d = 0; d < nd; d++) { const int L = olab[o][d]; if (L < base_ell) occ[L]++; } }
        for (int c = 0; c < 256; c++) {                                    /* once-only, ascending by character */
            const int L = lblmap[c]; if (L >= 0 && L < base_ell && occ[L] == 1) outlab[nout++] = L;
        }
    }

    /* output shape and strides */
    int64_t oshape[EINSUM_MAXLBL]; int64_t ostride_out[EINSUM_MAXLBL]; int64_t outsize;
    for (int k = 0; k < nout; k++) oshape[k] = lsize[outlab[k]] ? lsize[outlab[k]] : 1;
    { int64_t st = 1; for (int k = nout - 1; k >= 0; k--) { ostride_out[k] = st; st *= oshape[k]; } outsize = st; }

    /* total number of label combinations (guard against a runaway loop) */
    int64_t total = 1;
    for (int L = 0; L < nlab; L++) { const int64_t s = lsize[L] ? lsize[L] : 1; if (s && total > (int64_t)100000000 / s) { fn_set_error("einsum: contraction too large for this implementation"); return TSR_EARG; } total *= s; }

    /* materialise operands as contiguous float64 */
    double *data[EINSUM_MAXOP]; int64_t dn[EINSUM_MAXOP];
    for (int o = 0; o < nop; o++) { data[o] = fn_arg_doubles(&ops[o], &dn[o]); if (!data[o]) { for (int q = 0; q < o; q++) fn_free_doubles(data[q], dn[q]); fn_set_error("einsum: operand %d could not be read as float64", o + 1); return TSR_ENOMEM; } }

    double *out; double scalar = 0.0;
    if (nout == 0) { out = &scalar; }
    else { out = (double *)fn_result_array(&res[0], TSR_F64, (int32_t)nout, oshape); if (!out) { for (int o = 0; o < nop; o++) fn_free_doubles(data[o], dn[o]); return TSR_ENOMEM; } memset(out, 0, (size_t)outsize * sizeof(double)); }

    int64_t cur[EINSUM_MAXLBL]; for (int L = 0; L < nlab; L++) cur[L] = 0;
    for (int64_t it = 0; it < total; it++) {
        double prod = 1.0;
        for (int o = 0; o < nop && prod != 0.0; o++) {
            const int32_t nd = ops[o].arr.ndim; int64_t off = 0;
            for (int d = 0; d < nd; d++) { const int L = olab[o][d]; const int64_t i = (osz[o][d] == 1) ? 0 : cur[L]; off += i * ostr[o][d]; }
            prod *= data[o][off];
        }
        if (prod != 0.0) { int64_t oo = 0; for (int k = 0; k < nout; k++) oo += cur[outlab[k]] * ostride_out[k]; out[oo] += prod; }
        for (int L = nlab - 1; L >= 0; L--) { const int64_t s = lsize[L] ? lsize[L] : 1; if (++cur[L] < s) break; cur[L] = 0; }
    }

    for (int o = 0; o < nop; o++) fn_free_doubles(data[o], dn[o]);
    if (nout == 0) fn_result_num(&res[0], scalar);
    return TSR_OK;
}

/* finfo(dtype): the machine limits of a floating dtype (numpy.finfo), the common fields in the usual order.
   A complex dtype reports the limits of its real component, as NumPy does. */
static int r_finfo(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int dt, rc;
    if ((rc = arg_to_dtype(argk(args, nargs, 0), &dt)) < 0) return rc;
    if (dt == TSR_C128) dt = TSR_F64;
    double eps, epsneg, mx, tiny, resol;
    int prec, bits, nmant, nexp, maxexp, minexp;
    if (dt == TSR_F64) {
        eps = DBL_EPSILON; epsneg = DBL_EPSILON * 0.5; mx = DBL_MAX; tiny = DBL_MIN; resol = 1e-15;
        prec = 15; bits = 64; nmant = 52; nexp = 11; maxexp = 1024; minexp = -1022;
    } else if (dt == TSR_F32) {
        eps = (double)FLT_EPSILON; epsneg = (double)(FLT_EPSILON * 0.5f); mx = (double)FLT_MAX;
        tiny = (double)FLT_MIN; resol = (double)(float)1e-6;
        prec = 6; bits = 32; nmant = 23; nexp = 8; maxexp = 128; minexp = -126;
    } else {
        fn_set_error("finfo(dtype): a floating-point dtype is required"); return TSR_EARG;
    }
    fn_result_num(&res[0], eps);  fn_result_num(&res[1], epsneg); fn_result_num(&res[2], mx);
    fn_result_num(&res[3], -mx);  fn_result_num(&res[4], tiny);   fn_result_num(&res[5], tiny);
    fn_result_num(&res[6], resol);
    fn_result_int(&res[7], prec); fn_result_int(&res[8], bits);   fn_result_int(&res[9], nmant);
    fn_result_int(&res[10], nexp); fn_result_int(&res[11], maxexp); fn_result_int(&res[12], minexp);
    return TSR_OK;
}

/* iinfo(dtype): the limits of an integer dtype (numpy.iinfo) */
static int r_iinfo(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int dt, rc;
    if ((rc = arg_to_dtype(argk(args, nargs, 0), &dt)) < 0) return rc;
    int64_t mn, mx; int bits;
    switch (dt) {
        case TSR_I64: mn = INT64_MIN; mx = INT64_MAX; bits = 64; break;
        case TSR_I32: mn = INT32_MIN; mx = INT32_MAX; bits = 32; break;
        case TSR_U8:  mn = 0;         mx = 255;       bits = 8;  break;
        default: fn_set_error("iinfo(dtype): an integer dtype is required"); return TSR_EARG;
    }
    fn_result_int(&res[0], mn); fn_result_int(&res[1], mx); fn_result_int(&res[2], bits);
    return TSR_OK;
}

/* logsumexp(a, axis=None, keepdims=False): the stable log(sum(exp(a))) reduction (scipy.special.logsumexp).
   Real float64; the b weights and return_sign options are not supported here. */
static int r_logsumexp(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    harr ha;
    int rc = get_arr(argk(args, nargs, 0), &ha, "a");
    if (rc < 0) return rc;
    own src;
    if ((rc = own_from(&src, &ha.a, TSR_F64)) < 0) return rc;    /* contiguous float64 copy */
    const int32_t n = src.a.ndim;
    const double *d = (const double *)src.a.data;

    int keep = 0;
    if (!is_none(argk(args, nargs, 2))) { int64_t kv; if ((rc = int_param(argk(args, nargs, 2), "keepdims", &kv)) < 0) { own_free(&src); return rc; } keep = kv != 0; }

    const tsr_arg *axa = argk(args, nargs, 1);
    if (is_none(axa) || n == 0) {                                /* reduce the whole array to a scalar */
        const int64_t tot = tsr_shape_size(n, src.a.shape);
        double m = -INFINITY; for (int64_t i = 0; i < tot; i++) if (d[i] > m) m = d[i];
        double out;
        if (tot == 0 || m == -INFINITY) out = -INFINITY;
        else { double s = 0.0; for (int64_t i = 0; i < tot; i++) s += exp(d[i] - m); out = m + log(s); }
        own z;
        if (keep) { int64_t sh[TSR_MAXDIM]; for (int32_t k = 0; k < n; k++) sh[k] = 1; if ((rc = own_new(&z, TSR_F64, n, sh)) < 0) { own_free(&src); return rc; } }
        else if ((rc = own_new(&z, TSR_F64, 0, NULL)) < 0) { own_free(&src); return rc; }
        ((double *)z.a.data)[0] = out;
        own_free(&src);
        return own_emit(&res[0], &z);
    }

    int64_t ax; if ((rc = int_param(axa, "axis", &ax)) < 0) { own_free(&src); return rc; }
    if (ax < 0) ax += n;
    if (ax < 0 || ax >= n) { own_free(&src); fn_set_error("logsumexp: axis %lld is out of bounds for %d dimensions", (long long)ax, (int)n); return TSR_EARG; }
    const int64_t *shp = src.a.shape;
    int64_t L = shp[ax], outer = 1, inner = 1;
    for (int32_t k = 0; k < ax; k++) outer *= shp[k];
    for (int32_t k = ax + 1; k < n; k++) inner *= shp[k];

    int64_t osh[TSR_MAXDIM]; int32_t ond;
    if (keep) { ond = n; for (int32_t k = 0; k < n; k++) osh[k] = (k == ax) ? 1 : shp[k]; }
    else { ond = 0; for (int32_t k = 0; k < n; k++) if (k != ax) osh[ond++] = shp[k]; }
    own z;
    if ((rc = own_new(&z, TSR_F64, ond, osh)) < 0) { own_free(&src); return rc; }
    double *out = (double *)z.a.data;
    for (int64_t o = 0; o < outer; o++)
        for (int64_t i = 0; i < inner; i++) {
            const double *base = d + (o * L) * inner + i;
            double m = -INFINITY; for (int64_t j = 0; j < L; j++) { const double v = base[j * inner]; if (v > m) m = v; }
            double r;
            if (L == 0 || m == -INFINITY) r = -INFINITY;
            else { double s = 0.0; for (int64_t j = 0; j < L; j++) s += exp(base[j * inner] - m); r = m + log(s); }
            out[o * inner + i] = r;
        }
    own_free(&src);
    return own_emit(&res[0], &z);
}

/* softmax(x, axis=None) = exp(x - max) / sum(exp(x - max)) over the axis (scipy.special.softmax).
   log_softmax(x, axis=None) = (x - max) - log(sum(exp(x - max))). Both return a same-shape float64 array;
   the max/sum follow scipy's stabilised order (shared with r_logsumexp). */
static int softmax_impl(const tsr_arg *args, int nargs, tsr_result *res, int is_log)
{
    harr ha;
    int rc = get_arr(argk(args, nargs, 0), &ha, "x");
    if (rc < 0) return rc;
    own src;
    if ((rc = own_from(&src, &ha.a, TSR_F64)) < 0) return rc;
    const int32_t n = src.a.ndim;
    const double *d = (const double *)src.a.data;
    own z;
    if ((rc = own_new(&z, TSR_F64, n, src.a.shape)) < 0) { own_free(&src); return rc; }
    double *out = (double *)z.a.data;

    const tsr_arg *axa = argk(args, nargs, 1);
    if (is_none(axa) || n == 0) {                                /* over the whole (flattened) array */
        const int64_t tot = tsr_shape_size(n, src.a.shape);
        double m = -INFINITY; for (int64_t i = 0; i < tot; i++) if (d[i] > m) m = d[i];
        double s = 0.0; for (int64_t i = 0; i < tot; i++) s += exp(d[i] - m);
        const double ls = log(s);
        for (int64_t i = 0; i < tot; i++) out[i] = is_log ? (d[i] - m) - ls : exp(d[i] - m) / s;
        own_free(&src);
        return own_emit(&res[0], &z);
    }
    int64_t ax; if ((rc = int_param(axa, "axis", &ax)) < 0) { own_free(&src); own_free(&z); return rc; }
    if (ax < 0) ax += n;
    if (ax < 0 || ax >= n) { own_free(&src); own_free(&z);
        fn_set_error("softmax: axis %lld is out of bounds for %d dimensions", (long long)ax, (int)n); return TSR_EARG; }
    const int64_t *shp = src.a.shape;
    int64_t L = shp[ax], outer = 1, inner = 1;
    for (int32_t k = 0; k < ax; k++) outer *= shp[k];
    for (int32_t k = ax + 1; k < n; k++) inner *= shp[k];
    for (int64_t o = 0; o < outer; o++)
        for (int64_t i = 0; i < inner; i++) {
            const double *base = d + (o * L) * inner + i;
            double *ob = out + (o * L) * inner + i;
            double m = -INFINITY; for (int64_t j = 0; j < L; j++) { const double v = base[j * inner]; if (v > m) m = v; }
            double s = 0.0; for (int64_t j = 0; j < L; j++) s += exp(base[j * inner] - m);
            const double ls = log(s);
            for (int64_t j = 0; j < L; j++)
                ob[j * inner] = is_log ? (base[j * inner] - m) - ls : exp(base[j * inner] - m) / s;
        }
    own_free(&src);
    return own_emit(&res[0], &z);
}
static int r_softmax(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return softmax_impl(args, nargs, res, 0); }
static int r_log_softmax(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return softmax_impl(args, nargs, res, 1); }

/* fabs(x): the absolute value as float64, element-wise (numpy.fabs; always real float). */
static int r_fabs(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3) { fn_set_error("fabs: input must be an array"); return TSR_ETYPE; }
    int64_t n; double *x = fn_arg_doubles(&args[0], &n);
    if (!x) { fn_set_error("fabs: input could not be read as float64"); return TSR_ENOMEM; }
    double *out = (double *)fn_result_array(&res[0], TSR_F64, args[0].arr.ndim, args[0].arr.shape);
    if (!out) { fn_free_doubles(x, n); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) out[i] = fabs(x[i]);
    fn_free_doubles(x, n);
    return TSR_OK;
}

/* ===================================== completeness ufuncs: spacing, modf, bitwise_and/or/xor, fromiter, can_cast */

/* spacing(x): the signed distance from x to the adjacent float in the direction of x's sign (numpy.spacing) */
static int r_spacing(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    harr h; int rc = get_arr(argk(args, nargs, 0), &h, "x");
    if (rc < 0) return rc;
    own z;
    if ((rc = own_from(&z, &h.a, TSR_F64)) < 0) return rc;
    const int64_t n = tsr_shape_size(z.a.ndim, z.a.shape);
    double *p = (double *)z.a.data;
    for (int64_t i = 0; i < n; i++) {
        const double x = p[i];
        if (x != x) continue;                                /* NaN -> NaN */
        p[i] = nextafter(x, copysign((double)INFINITY, x)) - x;
    }
    return own_emit(res, &z);
}

/* modf(x): the fractional and integral parts as a (frac, intpart) pair, both carrying the sign of x (numpy.modf) */
static int r_modf(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    harr h; int rc = get_arr(argk(args, nargs, 0), &h, "x");
    if (rc < 0) return rc;
    own src;
    if ((rc = own_from(&src, &h.a, TSR_F64)) < 0) return rc;  /* contiguous float64 copy */
    const int64_t n = tsr_shape_size(src.a.ndim, src.a.shape);
    tsr_result *items = fn_result_seq(&res[0], 2);
    if (!items) { own_free(&src); return TSR_ENOMEM; }
    double *pf = (double *)fn_result_array(&items[0], TSR_F64, src.a.ndim, src.a.shape);
    double *pg = pf ? (double *)fn_result_array(&items[1], TSR_F64, src.a.ndim, src.a.shape) : NULL;
    if (!pf || !pg) { own_free(&src); return TSR_ENOMEM; }
    const double *s = (const double *)src.a.data;
    for (int64_t i = 0; i < n; i++) { const double t = trunc(s[i]); pg[i] = t; pf[i] = s[i] - t; }
    own_free(&src);
    return TSR_OK;
}

static const int BW_AND = 0, BW_OR = 1, BW_XOR = 2, BW_LSHIFT = 3, BW_RSHIFT = 4;

/* bitwise_and/or/xor(x1, x2): element-wise over integers; equal shapes or a scalar operand (numpy.bitwise_*) */
static int r_bitwise(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int op = ctx ? *(const int *)ctx : BW_AND;
    harr ha, hb;
    int rc = get_arr(argk(args, nargs, 0), &ha, "x1"); if (rc < 0) return rc;
    rc = get_arr(argk(args, nargs, 1), &hb, "x2"); if (rc < 0) return rc;
    if ((!is_intlike(ha.a.dtype) && ha.a.dtype != TSR_BOOL) || (!is_intlike(hb.a.dtype) && hb.a.dtype != TSR_BOOL)) {
        fn_set_error("bitwise operations are not supported for floating-point arguments");
        return TSR_ETYPE;
    }
    own a, b;
    if ((rc = own_from(&a, &ha.a, TSR_I64)) < 0) return rc;
    if ((rc = own_from(&b, &hb.a, TSR_I64)) < 0) { own_free(&a); return rc; }
    const int64_t na = tsr_shape_size(a.a.ndim, a.a.shape), nb = tsr_shape_size(b.a.ndim, b.a.shape);
    if (na != nb && na != 1 && nb != 1) { own_free(&a); own_free(&b); fn_set_error("operands could not be broadcast together"); return TSR_EARG; }
    const tsr_array *rsh = na >= nb ? &a.a : &b.a;
    own z;
    if ((rc = own_new(&z, TSR_I64, rsh->ndim, rsh->shape)) < 0) { own_free(&a); own_free(&b); return rc; }
    const int64_t *pa = (const int64_t *)a.a.data, *pb = (const int64_t *)b.a.data;
    int64_t *pz = (int64_t *)z.a.data;
    const int64_t nz = tsr_shape_size(z.a.ndim, z.a.shape);
    for (int64_t i = 0; i < nz; i++) {
        const int64_t va = pa[na == 1 ? 0 : i], vb = pb[nb == 1 ? 0 : i];
        int64_t r;
        switch (op) {
        case BW_OR:     r = va | vb; break;
        case BW_XOR:    r = va ^ vb; break;
        /* shifts: wrap via unsigned (defined) for left, arithmetic for right; out-of-range counts saturate like a
           two's-complement shift rather than invoking C undefined behaviour. */
        case BW_LSHIFT: r = (vb < 0 || vb >= 64) ? 0 : (int64_t)((uint64_t)va << vb); break;
        case BW_RSHIFT: r = (vb < 0) ? 0 : (vb >= 64) ? (va < 0 ? -1 : 0) : (va >> vb); break;
        default:        r = va & vb; break;  /* BW_AND */
        }
        pz[i] = r;
    }
    own_free(&a); own_free(&b);
    return own_emit(res, &z);
}

/* fromiter(iterable, dtype, count=-1): a 1-D array from the first count elements, cast to dtype (numpy.fromiter) */
static int r_fromiter(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const tsr_arg *d = argk(args, nargs, 0);
    if (d->kind != 3) { fn_set_error("fromiter: the iterable must be an array of numbers"); return TSR_EARG; }
    int dt, rc;
    if ((rc = arg_to_dtype(argk(args, nargs, 1), &dt)) < 0) return rc;
    const int64_t total = tsr_shape_size(d->arr.ndim, d->arr.shape);
    int64_t count = total;
    const tsr_arg *ca = argk(args, nargs, 2);
    if (!is_none(ca)) { int64_t c; if ((rc = int_param(ca, "count", &c)) < 0) return rc; if (c >= 0) count = c; }
    if (count > total) { fn_set_error("iterator too short: Expected %lld but iterator had only %lld items", (long long)count, (long long)total); return TSR_EARG; }
    own src;
    if ((rc = own_from(&src, &d->arr, dt)) < 0) return rc;    /* contiguous C-order copy in the target dtype */
    int64_t csh[1] = {count};
    own z;
    if ((rc = own_new(&z, dt, 1, csh)) < 0) { own_free(&src); return rc; }
    if (count > 0) memcpy(z.a.data, src.a.data, (size_t)count * tsr_itemsize(dt));
    own_free(&src);
    return own_emit(res, &z);
}

static int dt_kind(int dt)
{
    switch (dt) {
    case TSR_BOOL: return 0;
    case TSR_U8: return 1;
    case TSR_I32: case TSR_I64: return 2;
    case TSR_F32: case TSR_F64: return 3;
    case TSR_C128: return 4;
    default: return -1;
    }
}

/* whether a value of dtype `from` is representable without loss in `to` (numpy's 'safe' rule for our dtype set) */
static int cast_safe(int from, int to)
{
    if (from == to || from == TSR_BOOL) return 1;
    switch (from) {
    case TSR_U8:  return to == TSR_I32 || to == TSR_I64 || to == TSR_F32 || to == TSR_F64 || to == TSR_C128;
    case TSR_I32: return to == TSR_I64 || to == TSR_F64 || to == TSR_C128;
    case TSR_I64: return to == TSR_F64 || to == TSR_C128;     /* numpy treats int64 -> float64 as a safe cast */
    case TSR_F32: return to == TSR_F64 || to == TSR_C128;
    case TSR_F64: return to == TSR_C128;
    default: return 0;
    }
}

/* can_cast(from_, to, casting='safe'): whether the cast is allowed under the casting rule (numpy.can_cast) */
static int r_can_cast(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int from, to, rc;
    if ((rc = arg_to_dtype(argk(args, nargs, 0), &from)) < 0) return rc;
    if ((rc = arg_to_dtype(argk(args, nargs, 1), &to)) < 0) return rc;
    const char *mode = "safe";
    const tsr_arg *m = argk(args, nargs, 2);
    if (m->kind == 2 && m->str) mode = m->str;
    int ok;
    if (strcmp(mode, "unsafe") == 0) ok = 1;
    else if (strcmp(mode, "no") == 0 || strcmp(mode, "equiv") == 0) ok = from == to;
    else if (strcmp(mode, "same_kind") == 0) ok = cast_safe(from, to) || dt_kind(from) == dt_kind(to);
    else ok = cast_safe(from, to);                            /* 'safe' (the default) */
    memset(res, 0, sizeof *res);
    res->kind = 4;
    res->num = ok ? 1 : 0;
    return TSR_OK;
}

/* frexp(x): the normalised mantissa in [0.5, 1) and the integer exponent as a (mantissa, exponent) pair,
   so that x == mantissa * 2**exponent; exponent is int32 as in numpy.frexp (0/inf/nan give exponent 0). */
static int r_frexp(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    harr h; int rc = get_arr(argk(args, nargs, 0), &h, "x");
    if (rc < 0) return rc;
    own src;
    if ((rc = own_from(&src, &h.a, TSR_F64)) < 0) return rc;
    const int64_t n = tsr_shape_size(src.a.ndim, src.a.shape);
    tsr_result *items = fn_result_seq(&res[0], 2);
    if (!items) { own_free(&src); return TSR_ENOMEM; }
    double *pm = (double *)fn_result_array(&items[0], TSR_F64, src.a.ndim, src.a.shape);
    int32_t *pe = pm ? (int32_t *)fn_result_array(&items[1], TSR_I32, src.a.ndim, src.a.shape) : NULL;
    if (!pm || !pe) { own_free(&src); return TSR_ENOMEM; }
    const double *s = (const double *)src.a.data;
    for (int64_t i = 0; i < n; i++) { int e = 0; pm[i] = frexp(s[i], &e); pe[i] = (int32_t)e; }
    own_free(&src);
    return TSR_OK;
}

/* ldexp(x1, x2): x1 * 2**x2, with an integer exponent array (numpy.ldexp); broadcasts a scalar operand. */
static int r_ldexp(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    harr ha, hb;
    int rc = get_arr(argk(args, nargs, 0), &ha, "x1"); if (rc < 0) return rc;
    rc = get_arr(argk(args, nargs, 1), &hb, "x2"); if (rc < 0) return rc;
    own a, b;
    if ((rc = own_from(&a, &ha.a, TSR_F64)) < 0) return rc;
    if ((rc = own_from(&b, &hb.a, TSR_I64)) < 0) { own_free(&a); return rc; }
    const int64_t na = tsr_shape_size(a.a.ndim, a.a.shape), nb = tsr_shape_size(b.a.ndim, b.a.shape);
    if (na != nb && na != 1 && nb != 1) { own_free(&a); own_free(&b); fn_set_error("operands could not be broadcast together"); return TSR_EARG; }
    const tsr_array *rsh = na >= nb ? &a.a : &b.a;
    own z;
    if ((rc = own_new(&z, TSR_F64, rsh->ndim, rsh->shape)) < 0) { own_free(&a); own_free(&b); return rc; }
    const double *pa = (const double *)a.a.data; const int64_t *pb = (const int64_t *)b.a.data;
    double *pz = (double *)z.a.data;
    const int64_t nz = tsr_shape_size(z.a.ndim, z.a.shape);
    for (int64_t i = 0; i < nz; i++) {
        int64_t e = pb[nb == 1 ? 0 : i];
        if (e > INT_MAX) e = INT_MAX; else if (e < INT_MIN) e = INT_MIN;   /* ldexp takes int; clamp like numpy's overflow */
        pz[i] = ldexp(pa[na == 1 ? 0 : i], (int)e);
    }
    own_free(&a); own_free(&b);
    return own_emit(res, &z);
}

/* bitwise_count(x): the number of 1-bits in the absolute value of each integer element, as uint8 (numpy.bitwise_count). */
static int r_bitwise_count(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    harr h; int rc = get_arr(argk(args, nargs, 0), &h, "x");
    if (rc < 0) return rc;
    if (!is_intlike(h.a.dtype)) { fn_set_error("bitwise_count is not supported for floating-point arguments"); return TSR_ETYPE; }
    own a;
    if ((rc = own_from(&a, &h.a, TSR_I64)) < 0) return rc;
    own z;
    if ((rc = own_new(&z, TSR_U8, a.a.ndim, a.a.shape)) < 0) { own_free(&a); return rc; }
    const int64_t n = tsr_shape_size(a.a.ndim, a.a.shape);
    const int64_t *pa = (const int64_t *)a.a.data; uint8_t *pz = (uint8_t *)z.a.data;
    for (int64_t i = 0; i < n; i++) {
        uint64_t v = pa[i] < 0 ? -(uint64_t)pa[i] : (uint64_t)pa[i];
        pz[i] = (uint8_t)__builtin_popcountll(v);
    }
    own_free(&a);
    return own_emit(res, &z);
}

/* numpy's floor division and remainder as a consistent pair (npy_divmod): mod carries the sign of the divisor,
   floordiv is the matching quotient so that x1 == floordiv*x2 + mod. */
static double py_divmod(double a, double b, double *modulus)
{
    double div, mod, floordiv;
    mod = fmod(a, b);
    if (!b) { *modulus = mod; return a / b; }                /* div by zero -> nan/inf, mod is nan */
    div = (a - mod) / b;
    if (mod) { if ((b < 0) != (mod < 0)) { mod += b; div -= 1.0; } }
    else mod = copysign(0.0, b);
    if (div) { floordiv = floor(div); if (div - floordiv > 0.5) floordiv += 1.0; }
    else floordiv = copysign(0.0, a / b);
    *modulus = mod;
    return floordiv;
}

/* divmod(x1, x2): the element-wise (floor_divide, remainder) pair (numpy.divmod); integer or float, broadcasts a scalar. */
static int r_divmod(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    harr ha, hb;
    int rc = get_arr(argk(args, nargs, 0), &ha, "x1"); if (rc < 0) return rc;
    rc = get_arr(argk(args, nargs, 1), &hb, "x2"); if (rc < 0) return rc;
    const int ints = is_intlike(ha.a.dtype) && is_intlike(hb.a.dtype);
    const int dt = ints ? TSR_I64 : TSR_F64;
    own a, b;
    if ((rc = own_from(&a, &ha.a, dt)) < 0) return rc;
    if ((rc = own_from(&b, &hb.a, dt)) < 0) { own_free(&a); return rc; }
    const int64_t na = tsr_shape_size(a.a.ndim, a.a.shape), nb = tsr_shape_size(b.a.ndim, b.a.shape);
    if (na != nb && na != 1 && nb != 1) { own_free(&a); own_free(&b); fn_set_error("operands could not be broadcast together"); return TSR_EARG; }
    const tsr_array *rsh = na >= nb ? &a.a : &b.a;
    tsr_result *items = fn_result_seq(&res[0], 2);
    if (!items) { own_free(&a); own_free(&b); return TSR_ENOMEM; }
    void *pq = fn_result_array(&items[0], dt, rsh->ndim, rsh->shape);
    void *pr = pq ? fn_result_array(&items[1], dt, rsh->ndim, rsh->shape) : NULL;
    if (!pq || !pr) { own_free(&a); own_free(&b); return TSR_ENOMEM; }
    const int64_t nz = tsr_shape_size(rsh->ndim, rsh->shape);
    if (ints) {
        const int64_t *pa = (const int64_t *)a.a.data, *pb = (const int64_t *)b.a.data;
        int64_t *qq = (int64_t *)pq, *rr = (int64_t *)pr;
        for (int64_t i = 0; i < nz; i++) {
            const int64_t x = pa[na == 1 ? 0 : i], y = pb[nb == 1 ? 0 : i];
            if (y == 0) { qq[i] = 0; rr[i] = 0; continue; }   /* numpy integer div by zero -> 0, 0 */
            int64_t q = x / y, r = x % y;
            if (r != 0 && ((r < 0) != (y < 0))) { r += y; q -= 1; }
            qq[i] = q; rr[i] = r;
        }
    } else {
        const double *pa = (const double *)a.a.data, *pb = (const double *)b.a.data;
        double *qq = (double *)pq, *rr = (double *)pr;
        for (int64_t i = 0; i < nz; i++) {
            double m = 0.0;
            qq[i] = py_divmod(pa[na == 1 ? 0 : i], pb[nb == 1 ? 0 : i], &m);
            rr[i] = m;
        }
    }
    own_free(&a); own_free(&b);
    return TSR_OK;
}

/* ================================================================ table */

static const fn_def DEFS[] = {
    ROUTINE("np.finfo", 13, "dtype", "eps, epsneg, max, min, tiny, smallest_normal, resolution, precision, bits, nmant, nexp, maxexp, minexp", r_finfo, NULL, "The machine limits of a floating-point dtype as a dict of the common fields (numpy.finfo)."),
    ROUTINE("np.iinfo", 3, "dtype", "min, max, bits", r_iinfo, NULL, "The limits of an integer dtype as a dict (min, max, bits) (numpy.iinfo)."),
    ROUTINE("np.einsum", 1, "subscripts, *operands", "out", r_einsum, NULL, "Einstein summation over the operands (numpy.einsum); real float64 operands."),
    ROUTINE("np.result_type", 1, "*arrays_and_dtypes", "out", r_result_type, NULL, "The dtype resulting from NumPy's type-promotion rules (numpy.result_type)."),
    ROUTINE("np.promote_types", 1, "type1, type2", "out", r_promote_types, NULL, "The smallest dtype both arguments can be safely cast to (numpy.promote_types)."),
    ROUTINE("np.round", 1, "a, decimals=0", "out", r_round, NULL, "Round to the given number of decimals, half to even (numpy.round)."),
    ROUTINE("np.around", 1, "a, decimals=0", "out", r_round, NULL, "Round to the given number of decimals, half to even (numpy.around; an alias of round)."),
    ROUTINE("np.fabs", 1, "x", "out", r_fabs, NULL, "Element-wise absolute value as float64 (numpy.fabs)."),
    ROUTINE("special.logsumexp", 1, "a, axis=None, keepdims=False", "out", r_logsumexp, NULL, "Stable log of the sum of exponentials over an axis (scipy.special.logsumexp; real a, no b/return_sign)."),
    ROUTINE("special.softmax", 1, "x, axis=None", "out", r_softmax, NULL, "exp(x) normalised over an axis (scipy.special.softmax; real)."),
    ROUTINE("special.logSoftmax", 1, "x, axis=None", "out", r_log_softmax, NULL, "log of softmax over an axis (scipy.special.log_softmax; real)."),
    ROUTINE("np.isclose", 1, "a, b, rtol=1e-05, atol=1e-08, equal_nan=False", "out", r_isclose, NULL, "Element-wise test for approximate equality within a tolerance (numpy.isclose)."),
    ROUTINE("np.allclose", 1, "a, b, rtol=1e-05, atol=1e-08, equal_nan=False", "out", r_allclose, NULL, "True when two arrays are element-wise equal within a tolerance (numpy.allclose)."),
    ROUTINE("np.outer", 1, "a, b", "out", r_outer, NULL, "Outer product of two vectors (numpy.outer)."),
    ROUTINE("np.inner", 1, "a, b", "out", r_inner, NULL, "Inner product over the last axes (numpy.inner)."),
    ROUTINE("np.vdot", 1, "a, b", "out", r_vdot, NULL, "Dot product of the flattened arrays (numpy.vdot)."),
    ROUTINE("np.kron", 1, "a, b", "out", r_kron, NULL, "Kronecker product (numpy.kron)."),
    ROUTINE("np.cross", 1, "a, b, axisa=-1, axisb=-1, axisc=-1, axis=None", "out", r_cross, NULL, "Cross product of 3- (or 2-) element vectors (numpy.cross)."),
    ROUTINE("np.tensordot", 1, "a, b, axes=2", "out", r_tensordot, NULL, "Tensor dot product over the given axes (numpy.tensordot)."),
    ROUTINE("np.diff", 1, "a, n=1, axis=-1, prepend=None, append=None", "out", r_diff, NULL, "n-th discrete difference along an axis (numpy.diff)."),
    ROUTINE("np.ediff1d", 1, "ary, to_end=None, to_begin=None", "out", r_ediff1d, NULL, "Differences between consecutive elements of the flattened array (numpy.ediff1d)."),
    ROUTINE("np.gradient", 1, "f, *varargs, axis=None, edge_order=1", "out", r_gradient, NULL, "Gradient by central differences, one-sided at the edges (numpy.gradient); several axes give a list."),
    ROUTINE("np.trapezoid", 1, "y, x=None, dx=1.0, axis=-1", "out", r_trapezoid, NULL, "Integral by the composite trapezoidal rule (numpy.trapezoid)."),
    ROUTINE("np.cumulativeTrapezoid", 1, "y, x=None, dx=1.0, axis=-1, initial=None", "out", r_cumulative_trapezoid, NULL, "Running integral by the trapezoidal rule (scipy.integrate.cumulative_trapezoid)."),
    ROUTINE("np.simpson", 1, "y, x=None, dx=1.0, axis=-1", "out", r_simpson, NULL, "Integral by the composite Simpson's rule (scipy.integrate.simpson)."),
    ROUTINE("np.unwrap", 1, "p, discont=None, axis=-1, period=6.283185307179586", "out", r_unwrap, NULL, "Unwrap by taking the complement of large jumps with respect to the period (numpy.unwrap)."),
    ROUTINE("np.interp", 1, "x, xp, fp, left=None, right=None, period=None", "out", r_interp, NULL, "One-dimensional piecewise linear interpolation (numpy.interp)."),
    ROUTINE("np.logspace", 1, "start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0", "out", r_logspace, NULL, "Numbers spaced evenly on a log scale (numpy.logspace)."),
    ROUTINE("np.geomspace", 1, "start, stop, num=50, endpoint=True, dtype=None, axis=0", "out", r_geomspace, NULL, "Numbers spaced evenly on a log scale, a geometric progression (numpy.geomspace)."),
    ROUTINE("np.vander", 1, "x, N=None, increasing=False", "out", r_vander, NULL, "Vandermonde matrix (numpy.vander)."),
    ROUTINE("np.polyval", 1, "p, x", "out", r_polyval, NULL, "Evaluate a polynomial at x (numpy.polyval)."),
    ROUTINE("np.polyadd", 1, "a1, a2", "out", r_polyaddsub, &PADD, "Sum of two polynomials (numpy.polyadd)."),
    ROUTINE("np.polysub", 1, "a1, a2", "out", r_polyaddsub, &PSUB, "Difference of two polynomials (numpy.polysub)."),
    ROUTINE("np.polymul", 1, "a1, a2", "out", r_polymul, NULL, "Product of two polynomials (numpy.polymul)."),
    ROUTINE("np.polyder", 1, "p, m=1", "out", r_polyder, NULL, "Derivative of a polynomial (numpy.polyder)."),
    ROUTINE("np.polyint", 1, "p, m=1, k=None", "out", r_polyint, NULL, "Antiderivative of a polynomial (numpy.polyint)."),
    ROUTINE("np.polydiv", 1, "u, v", "*quotient_remainder", r_polydiv, NULL, "Quotient and remainder of polynomial division (numpy.polydiv)."),
    ROUTINE("np.bartlett", 1, "M", "out", r_window, &WB, "Bartlett (triangular) window (numpy.bartlett)."),
    ROUTINE("np.blackman", 1, "M", "out", r_window, &WBL, "Blackman window (numpy.blackman)."),
    ROUTINE("np.hamming", 1, "M", "out", r_window, &WHM, "Hamming window (numpy.hamming)."),
    ROUTINE("np.hanning", 1, "M", "out", r_window, &WHN, "Hanning window (numpy.hanning)."),
    ROUTINE("np.kaiser", 1, "M, beta", "out", r_kaiser, NULL, "Kaiser window (numpy.kaiser)."),
    ROUTINE("np.i0", 1, "x", "out", r_i0, NULL, "Modified Bessel function of the first kind, order 0 (numpy.i0)."),
    ROUTINE("np.sinc", 1, "x", "out", r_sinc, NULL, "Normalised sinc, sin(pi x) / (pi x) (numpy.sinc)."),
    ROUTINE("np.nan_to_num", 1, "x, copy=True, nan=0.0, posinf=None, neginf=None", "out", r_nan_to_num, NULL, "Replace NaN and infinities by finite numbers (numpy.nan_to_num)."),
    ROUTINE("np.isposinf", 1, "x", "out", r_isinf_sign, &POS, "Element-wise test for positive infinity (numpy.isposinf)."),
    ROUTINE("np.isneginf", 1, "x", "out", r_isinf_sign, &NEG, "Element-wise test for negative infinity (numpy.isneginf)."),
    ROUTINE("np.iscomplex", 1, "x", "out", r_complex_pred, &P_ISCOMPLEX, "Element-wise test for a non-zero imaginary part (numpy.iscomplex)."),
    ROUTINE("np.isreal", 1, "x", "out", r_complex_pred, &P_ISREAL, "Element-wise test for a zero imaginary part (numpy.isreal)."),
    ROUTINE("np.iscomplexobj", 1, "x", "out", r_complex_pred, &P_ISCOMPLEXOBJ, "True for a complex dtype (numpy.iscomplexobj)."),
    ROUTINE("np.isrealobj", 1, "x", "out", r_complex_pred, &P_ISREALOBJ, "True for a non-complex dtype (numpy.isrealobj)."),
    ROUTINE("np.real_if_close", 1, "a, tol=100", "out", r_real_if_close, NULL, "The real part when every imaginary part is close to zero (numpy.real_if_close)."),
    ROUTINE("np.pad", 1, "array, pad_width, mode='constant', constant_values=None, end_values=None, stat_length=None, reflect_type=None", "out", r_pad, NULL,
            "Pad an array (numpy.pad): constant, edge, linear_ramp, maximum, mean, median, minimum, reflect, symmetric, wrap, empty."),
    ROUTINE("np.put", 0, "&a, ind, v, mode='raise'", "", r_put, NULL, "Write values at flat indices, in place (numpy.put)."),
    ROUTINE("np.putmask", 0, "&a, mask, values", "", r_putmask, &PUTMASK, "Write values where the mask holds, in place (numpy.putmask)."),
    ROUTINE("np.place", 0, "&arr, mask, vals", "", r_putmask, &PLACE, "Write successive values where the mask holds, in place (numpy.place)."),
    ROUTINE("np.copyto", 0, "&dst, src, casting='same_kind', where=None", "", r_copyto, NULL, "Copy values into an array, broadcasting, in place (numpy.copyto)."),
    ROUTINE("np.fill_diagonal", 0, "&a, val, wrap=False", "", r_fill_diagonal, NULL, "Fill the main diagonal, in place (numpy.fill_diagonal)."),
    ROUTINE("np.put_along_axis", 0, "&arr, indices, values, axis", "", r_put_along_axis, NULL, "Write values at matching 1-D index slices, in place (numpy.put_along_axis)."),
    ROUTINE("np.spacing", 1, "x", "out", r_spacing, NULL, "The signed distance from each element to the adjacent representable float (numpy.spacing)."),
    ROUTINE("np.modf", 1, "x", "*frac_intpart", r_modf, NULL, "The fractional and integral parts of each element as a (frac, intpart) pair, both carrying the sign of x (numpy.modf)."),
    ROUTINE("np.frexp", 1, "x", "*mantissa_exponent", r_frexp, NULL, "The normalised mantissa and integer exponent of each element as a (mantissa, exponent) pair (numpy.frexp)."),
    ROUTINE("np.ldexp", 1, "x1, x2", "out", r_ldexp, NULL, "x1 * 2**x2 element-wise, with an integer exponent (numpy.ldexp)."),
    ROUTINE("np.divmod", 1, "x1, x2", "*quotient_remainder", r_divmod, NULL, "The element-wise floor-division quotient and remainder as a pair (numpy.divmod)."),
    ROUTINE("np.bitwise_count", 1, "x", "out", r_bitwise_count, NULL, "The number of 1-bits in the absolute value of each integer element, as uint8 (numpy.bitwise_count)."),
    ROUTINE("np.bitwise_and", 1, "x1, x2", "out", r_bitwise, &BW_AND, "Element-wise bitwise AND of two integer arrays (numpy.bitwise_and)."),
    ROUTINE("np.bitwise_or", 1, "x1, x2", "out", r_bitwise, &BW_OR, "Element-wise bitwise OR of two integer arrays (numpy.bitwise_or)."),
    ROUTINE("np.bitwise_xor", 1, "x1, x2", "out", r_bitwise, &BW_XOR, "Element-wise bitwise XOR of two integer arrays (numpy.bitwise_xor)."),
    ROUTINE("np.left_shift", 1, "x1, x2", "out", r_bitwise, &BW_LSHIFT, "Shift the bits of each integer left by the per-element count (numpy.left_shift)."),
    ROUTINE("np.right_shift", 1, "x1, x2", "out", r_bitwise, &BW_RSHIFT, "Shift the bits of each integer right by the per-element count (numpy.right_shift)."),
    ROUTINE("np.fromiter", 1, "iterable, dtype, count=-1", "out", r_fromiter, NULL, "A 1-D array built from the first count elements of an iterable, cast to dtype (numpy.fromiter)."),
    ROUTINE("np.can_cast", 1, "from_, to, casting='safe'", "out", r_can_cast, NULL, "Whether a cast between two dtypes is allowed under a casting rule (numpy.can_cast)."),
};

const fn_table TSR_NP_NUMERIC_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
