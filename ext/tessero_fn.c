/*
 * Tessero\Ext\Special, Tessero\Ext\Stats, Tessero\Ext\Statistics: the function registry (ADR 0011).
 *
 * libtessero describes every registry function (tsr_fn_info: kind, argument/output/parameter names,
 * defaults, enums). At start-up this file reads those descriptions, registers one static method per function
 * with matching arginfo (so named arguments and reflection work), and routes every call to one of two
 * generic handlers: element-wise functions (ufuncs, distribution methods) and generalised ufuncs. The FFI
 * package interprets the same descriptions (Tessero\Native\Registry), so both backends expose the same names
 * and return identical bytes.
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "Zend/zend_exceptions.h"
#include "ext/json/php_json.h"
#include "libtessero/src/internal.h"
#define TSR_TESSERO_H_INCLUDED
#include "php_tessero.h"

#include <math.h>
#include <string.h>

enum { D_F64 = 0, D_F32 = 1, D_I64 = 2, D_I32 = 3, D_U8 = 4, D_BOOL = 5, D_C128 = 6 };
enum { K_UFUNC = 1, K_GUFUNC = 2, K_DIST = 3, K_ROUTINE = 4, K_RANDOM = 5 };

#define MAXP 16
#define MAXE 16

typedef struct {
    char name[40];
    int is_enum;
    int is_bool;   /* default was True/False: shown as a PHP bool */
    int nenum;
    char enums[MAXE][32];
    double def;                     /* default for numeric parameters */
} fn_param;

typedef struct {
    int id;
    int kind;
    int nin, nout;
    unsigned int_args;
    int has_int;
    char outs[16][24];
    int nparams;
    fn_param params[MAXP];
    int axis_none;                  /* default axis None */
    int axis_default;
    int axis_hidden;   /* no axis/keepdims arguments (digitize, isin, searchsorted) */
    int axis_inputs;
    int out_int;
    int out_bool;
    int out_int_like;  /* bit o: output o is int64 when input 0 is an integer array */
    int keep_param;    /* output 0 keeps an integer or bool input 0's dtype when bit params[keep_param] of keep_mask is set */
    int keep_mask;
    int axis_single;   /* axis is an int or null (a list is a TypeError) */
    int keepdims_ok;   /* takes keepdims */
    unsigned optional;              /* bit k: input k may be null (gufunc) */
    double in_def[4];               /* numeric default of optional input k (NaN: None, the kernel's placeholder) */
    char label[64];
    int discrete;
    int nshape;
    /* routines */
    unsigned seq;                   /* bit k: argument k is a sequence (kind 5) */
    unsigned inplace;               /* bit k: argument k is written in place (must be a writable NDArray) */
    int variadic;                   /* index of the variadic argument (numpy's *xi), -1 none */
    int out_seq;                    /* the result is one sequence (kind 6): a PHP list */
    int nrargs;                     /* argument names/defaults below (variadic routines) */
    char *rnames[32];               /* camelCase names */
    char *rdefs[32];                /* registry default ("None", "True", "'xy'", "10"), NULL: required */
} fn_bind;

static fn_bind *g_binds = NULL;
static int g_nbinds = 0;
static HashTable g_by_method;       /* "class::method" (lower case) -> bind index */

zend_class_entry *tsr_ce_special, *tsr_ce_stats, *tsr_ce_np, *tsr_ce_linalg, *tsr_ce_slinalg, *tsr_ce_distance, *tsr_ce_sparse, *tsr_ce_signal, *tsr_ce_ndimage, *tsr_ce_csgraph, *tsr_ce_signal_windows;

/* ================================================================ helpers */

static int is_ndarray(zval *z) { return Z_TYPE_P(z) == IS_OBJECT && instanceof_function(Z_OBJCE_P(z), tsr_ce_ndarray); }
static int is_scalar(zval *z) { return Z_TYPE_P(z) == IS_LONG || Z_TYPE_P(z) == IS_DOUBLE || Z_TYPE_P(z) == IS_TRUE || Z_TYPE_P(z) == IS_FALSE; }

static double zval_num(zval *z)
{
    switch (Z_TYPE_P(z)) {
    case IS_LONG: return (double)Z_LVAL_P(z);
    case IS_DOUBLE: return Z_DVAL_P(z);
    case IS_TRUE: return 1.0;
    case IS_FALSE: return 0.0;
    default: return NAN;
    }
}

static void throw_fn(int rc, const char *label)
{
    const char *msg = tsr_fn_error();
    if (msg && *msg) {
        /* the kernel's codes, as tsr_throw_rc maps them; an argument error is PHP's ValueError */
        zend_class_entry *ce = rc == -6 ? tsr_ce_shape_exception : rc == -3 ? tsr_ce_memory_exception
                             : rc == -2 ? tsr_ce_index_exception : rc == -4 ? tsr_ce_dtype_exception : zend_ce_value_error;
        zend_throw_exception_ex(ce, 0, "%s: %s", label, msg);
        return;
    }
    tsr_throw_rc(rc, label);
}

static void camel(const char *in, char *out, size_t cap)
{
    size_t k = 0;
    int up = 0;
    for (; *in && k + 1 < cap; in++) {
        /* the rule of tools/parity/gen-facades.php camel(): an empty part (a trailing or doubled underscore)
           stays an underscore, so SciPy's lambda_ is lambda_ on both backends */
        if (*in == '_') {
            if (in[1] == '_' || in[1] == '\0') out[k++] = '_';
            else up = 1;
            continue;
        }
        out[k++] = up ? (char)toupper((unsigned char)*in) : *in;
        up = 0;
    }
    out[k] = '\0';
}

/* ================================================================ element-wise handler */

static void call_ufunc(const fn_bind *b, int method, zval *args, int nargs, zval *out, zval *rv)
{
    int nin = b->nin, nout = b->nout;
    const char *outs_dist_stats[4] = {"mean", "var", "skew", "kurtosis"};
    const char *outs_support[2] = {"a", "b"};
    if (b->kind == K_DIST) {
        if (tsr_fn_arity(b->id, method, &nin, &nout) != 0) { zend_throw_error(NULL, "%s: bad method", b->label); return; }
    }
    if (nargs != nin) {
        zend_argument_count_error("%s() takes %d argument(s), %d given", b->label, nin, nargs);
        return;
    }
    int kmethod = b->kind == K_DIST ? method : 0;
    if (b->kind == K_UFUNC && b->has_int) {
        int ints = 1;
        for (int k = 0; k < nin; k++) {
            if (!(b->int_args & (1u << k))) continue;
            zval *z = &args[k];
            ZVAL_DEREF(z);
            if (Z_TYPE_P(z) == IS_LONG || Z_TYPE_P(z) == IS_TRUE || Z_TYPE_P(z) == IS_FALSE) continue;
            if (is_ndarray(z)) {
                const int dt = Z_TSR_P(z)->a.dtype;
                if (dt == D_I64 || dt == D_I32 || dt == D_U8 || dt == D_BOOL) continue;
            }
            ints = 0;
            break;
        }
        if (ints) kmethod = 1;
    }
    /* scalar fast path */
    int all_scalar = out == NULL || Z_TYPE_P(out) == IS_NULL;
    for (int k = 0; k < nin && all_scalar; k++) {
        zval *z = &args[k];
        ZVAL_DEREF(z);
        if (!is_scalar(z)) all_scalar = 0;
    }
    if (all_scalar) {
        double v[MAXE + 4];
        void *ptrs[MAXE + 4];
        for (int k = 0; k < nin; k++) { zval *z = &args[k]; ZVAL_DEREF(z); v[k] = zval_num(z); }
        for (int k = 0; k < nin + nout; k++) ptrs[k] = &v[k];
        const int rc = tsr_fn_ufunc(b->id, kmethod, D_F64, 0, NULL, nin + nout, ptrs, NULL);
        if (rc < 0) { throw_fn(rc, b->label); return; }
        if (nout == 1) { ZVAL_DOUBLE(rv, v[nin]); return; }
        array_init_size(rv, (uint32_t)nout);
        for (int k = 0; k < nout; k++) {
            const char *nm = b->kind == K_DIST ? (nout == 4 ? outs_dist_stats[k] : outs_support[k]) : b->outs[k];
            add_assoc_double(rv, nm, v[nin + k]);
        }
        return;
    }
    /* arrays */
    zval ops[MAXE];
    int nops = 0, f32 = 1, any_array = 0, any_complex = 0;
    for (int k = 0; k < nin; k++) {
        zval *z = &args[k];
        ZVAL_DEREF(z);
        if (is_ndarray(z)) { any_array = 1; const int d = Z_TSR_P(z)->a.dtype; if (d == D_C128) any_complex = 1; else if (d != D_F32) f32 = 0; }
        else if (Z_TYPE_P(z) == IS_ARRAY) { any_array = 1; f32 = 0; }
    }
    /* complex128 operands run the ufunc's complex loop if it has one (else tsr_fn_ufunc returns a DType error) */
    const int dt = any_complex ? D_C128 : ((any_array && f32) ? D_F32 : D_F64);
    int ok = 1;
    for (int k = 0; k < nin; k++) {
        zval *z = &args[k], tmp;
        ZVAL_DEREF(z);
        if (tsr_ndarray_from_zval(z, is_ndarray(z) ? -1 : D_F64, &tmp) == FAILURE) { ok = 0; break; }
        if (tsr_ndarray_contiguous_as(&tmp, dt, &ops[nops]) == FAILURE) { zval_ptr_dtor(&tmp); ok = 0; break; }
        zval_ptr_dtor(&tmp);
        nops++;
    }
    if (!ok) { for (int k = 0; k < nops; k++) zval_ptr_dtor(&ops[k]); return; }
    /* broadcast shape */
    int32_t nd = 0;
    int64_t shape[32];
    for (int k = 0; k < nin; k++) {
        const tsr_array *a = &Z_TSR(ops[k])->a;
        int32_t rnd;
        int64_t rsh[32];
        if (k == 0) { nd = a->ndim; memcpy(shape, a->shape, sizeof(int64_t) * (size_t)a->ndim); continue; }
        if (tsr_broadcast_shape(nd, shape, a->ndim, a->shape, &rnd, rsh) != 0) {
            zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s(): operands could not be broadcast together", b->label);
            for (int j = 0; j < nops; j++) zval_ptr_dtor(&ops[j]);
            return;
        }
        nd = rnd;
        memcpy(shape, rsh, sizeof(int64_t) * (size_t)nd);
    }
    zval res[4];
    for (int k = 0; k < nout; k++) {
        if (tsr_ndarray_new(&res[k], dt, nd, shape, 0) == FAILURE) {
            for (int j = 0; j < k; j++) zval_ptr_dtor(&res[j]);
            for (int j = 0; j < nops; j++) zval_ptr_dtor(&ops[j]);
            return;
        }
    }
    int64_t n = 1;
    for (int d = 0; d < nd; d++) n *= shape[d];
    int rc = 0;
    if (n > 0) {
        void *ptrs[MAXE + 4];
        int64_t *strides = emalloc(sizeof(int64_t) * (size_t)((nin + nout) * (nd > 0 ? nd : 1)));
        for (int k = 0; k < nin; k++) {
            const tsr_array *a = &Z_TSR(ops[k])->a;
            ptrs[k] = (char *)a->data + a->offset;
            if (nd > 0) tsr_broadcast_strides(a, nd, shape, strides + (size_t)k * nd);
        }
        for (int k = 0; k < nout; k++) {
            const tsr_array *a = &Z_TSR(res[k])->a;
            ptrs[nin + k] = (char *)a->data + a->offset;
            if (nd > 0) memcpy(strides + (size_t)(nin + k) * nd, a->strides, sizeof(int64_t) * (size_t)nd);
        }
        rc = tsr_fn_ufunc(b->id, kmethod, dt, nd, shape, nin + nout, ptrs, nd > 0 ? strides : NULL);
        efree(strides);
    }
    for (int k = 0; k < nops; k++) zval_ptr_dtor(&ops[k]);
    if (rc < 0) {
        for (int k = 0; k < nout; k++) zval_ptr_dtor(&res[k]);
        throw_fn(rc, b->label);
        return;
    }
    /* out: write the results into the given array(s) */
    if (out && Z_TYPE_P(out) != IS_NULL) {
        zval *targets[4];
        int nt = 0;
        if (is_ndarray(out)) targets[nt++] = out;
        else if (Z_TYPE_P(out) == IS_ARRAY) {
            zval *e;
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(out), e) { if (nt < 4) targets[nt++] = e; } ZEND_HASH_FOREACH_END();
        }
        if (nt != nout) {
            for (int k = 0; k < nout; k++) zval_ptr_dtor(&res[k]);
            zend_argument_count_error("%s(): out must hold %d array(s)", b->label, nout);
            return;
        }
        for (int k = 0; k < nout; k++) {
            if (!is_ndarray(targets[k])) {
                for (int j = 0; j < nout; j++) zval_ptr_dtor(&res[j]);
                zend_type_error("%s(): out must be Tessero\\Ext\\NDArray", b->label);
                return;
            }
            tsr_obj *dst = Z_TSR_P(targets[k]);
            if (tsr_require_writable(dst, b->label) == FAILURE || tsr_cast_into(&Z_TSR(res[k])->a, dst, b->label) == FAILURE) {
                for (int j = 0; j < nout; j++) zval_ptr_dtor(&res[j]);
                return;
            }
            zval_ptr_dtor(&res[k]);
            ZVAL_COPY(&res[k], targets[k]);
        }
    }
    if (nout == 1) { ZVAL_COPY_VALUE(rv, &res[0]); return; }
    array_init_size(rv, (uint32_t)nout);
    for (int k = 0; k < nout; k++) {
        const char *nm = b->kind == K_DIST ? (nout == 4 ? outs_dist_stats[k] : outs_support[k]) : b->outs[k];
        add_assoc_zval(rv, nm, &res[k]);
    }
}

/* ================================================================ generalised ufunc handler */

static int axis_mask(zval *axis, int nd, uint64_t *mask, const char *label)
{
    *mask = 0;
    if (axis == NULL || Z_TYPE_P(axis) == IS_NULL) { *mask = nd >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << nd) - 1); return SUCCESS; }
    zval one, *e;
    HashTable *list = NULL;
    if (Z_TYPE_P(axis) == IS_LONG) { ZVAL_COPY_VALUE(&one, axis); }
    else if (Z_TYPE_P(axis) == IS_ARRAY) list = Z_ARRVAL_P(axis);
    else { zend_type_error("%s(): axis must be int, list of int or null", label); return FAILURE; }
    if (!list) {
        zend_long a = Z_LVAL(one);
        if (a < 0) a += nd;
        if (a < 0 || a >= nd) { zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s(): axis " ZEND_LONG_FMT " is out of bounds for an array of dimension %d", label, Z_LVAL(one), nd); return FAILURE; }
        *mask = (uint64_t)1 << a;
        return SUCCESS;
    }
    ZEND_HASH_FOREACH_VAL(list, e) {
        if (Z_TYPE_P(e) != IS_LONG) { zend_type_error("%s(): axis must be int, list of int or null", label); return FAILURE; }
        zend_long a = Z_LVAL_P(e);
        if (a < 0) a += nd;
        if (a < 0 || a >= nd) { zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s(): axis " ZEND_LONG_FMT " is out of bounds for an array of dimension %d", label, Z_LVAL_P(e), nd); return FAILURE; }
        if (*mask & ((uint64_t)1 << a)) { zend_value_error("%s(): repeated axis", label); return FAILURE; }
        *mask |= (uint64_t)1 << a;
    } ZEND_HASH_FOREACH_END();
    return SUCCESS;
}

static int param_values(const fn_bind *b, zval *given, int ngiven, double *out)
{
    for (int k = 0; k < b->nparams; k++) {
        const fn_param *p = &b->params[k];
        zval *z = k < ngiven ? &given[k] : NULL;
        if (z) ZVAL_DEREF(z);
        if (p->is_enum) {
            if (!z || Z_TYPE_P(z) == IS_NULL || Z_TYPE_P(z) == IS_UNDEF) { out[k] = 0; continue; }
            if (Z_TYPE_P(z) != IS_STRING) { zend_type_error("%s(): %s must be a string", b->label, p->name); return FAILURE; }
            int found = -1;
            for (int e = 0; e < p->nenum; e++) if (strcmp(Z_STRVAL_P(z), p->enums[e]) == 0) { found = e; break; }
            if (found < 0) { zend_value_error("%s(): %s must be one of the documented values, got '%s'", b->label, p->name, Z_STRVAL_P(z)); return FAILURE; }
            out[k] = found;
            continue;
        }
        if (!z || Z_TYPE_P(z) == IS_UNDEF) { out[k] = p->def; continue; }
        if (Z_TYPE_P(z) == IS_NULL) { out[k] = NAN; continue; }
        if (!is_scalar(z)) { zend_type_error("%s(): %s must be a number, bool or null", b->label, p->name); return FAILURE; }
        out[k] = zval_num(z);
    }
    return SUCCESS;
}

static void call_gufunc(const fn_bind *b, zval *args, int nargs, zval *rv)
{
    /* argument order: arrays..., axis, params..., keepdims (arrays..., params... when the axis is hidden) */
    const int nin = b->nin;
    const int ax = b->axis_hidden ? 0 : 1;
    int nreq = nin;
    while (nreq > 0 && (b->optional & (1u << (nreq - 1)))) nreq--;
    if (nargs < nreq) { zend_argument_count_error("%s() takes at least %d argument(s)", b->label, nreq); return; }
    zval arrs[4];
    int n = 0;
    for (int k = 0; k < nin; k++) {
        zval undef;
        ZVAL_UNDEF(&undef);
        zval *z = k < nargs ? &args[k] : &undef;
        ZVAL_DEREF(z);
        if ((b->optional & (1u << k)) && (Z_TYPE_P(z) == IS_UNDEF || (Z_TYPE_P(z) == IS_NULL && isnan(b->in_def[k])))) {
            /* an omitted optional input: its registry default (circmean's high = 2 pi, kstat's n = 2) as a 0-d
               array, or for a None default (weights) the kernel's placeholder 0-d 1.0, as the FFI façade passes */
            if (tsr_ndarray_new(&arrs[n], D_F64, 0, NULL, 0) == FAILURE) { for (int j = 0; j < n; j++) zval_ptr_dtor(&arrs[j]); return; }
            *(double *)((char *)Z_TSR(arrs[n])->a.data + Z_TSR(arrs[n])->a.offset) = isnan(b->in_def[k]) ? 1.0 : b->in_def[k];
            n++;
            continue;
        }
        if (tsr_ndarray_from_zval(z, -1, &arrs[n]) == FAILURE) { for (int j = 0; j < n; j++) zval_ptr_dtor(&arrs[j]); return; }
        if (Z_TSR(arrs[n])->a.dtype == D_C128) {
            for (int j = 0; j <= n; j++) zval_ptr_dtor(&arrs[j]);
            zend_throw_exception_ex(tsr_ce_dtype_exception, 0, "%s() does not support complex128 input", b->label);
            return;
        }
        n++;
    }
    zval *axis = ax && nargs > nin ? &args[nin] : NULL;
    zval axis_default;
    if (axis == NULL || Z_TYPE_P(axis) == IS_UNDEF) {
        if (b->axis_none) ZVAL_NULL(&axis_default); else ZVAL_LONG(&axis_default, b->axis_default);
        axis = &axis_default;
    }
    ZVAL_DEREF(axis);
    if (b->axis_single && Z_TYPE_P(axis) == IS_ARRAY) {
        zend_type_error("%s(): axis must be an integer or null, not a list of axes", b->label);
        goto fail;
    }
    tsr_array in[4];
    uint64_t masks[4];
    for (int k = 0; k < nin; k++) {
        in[k] = Z_TSR(arrs[k])->a;
        const int whole = k > 0 && (in[k].ndim == 0 || (in[k].ndim == 1 && Z_TSR(arrs[0])->a.ndim > 1));
        if (k < b->axis_inputs && !whole) {
            if (axis_mask(axis, in[k].ndim, &masks[k], b->label) == FAILURE) goto fail;
        } else {
            masks[k] = in[k].ndim >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << in[k].ndim) - 1);
        }
    }
    double pv[MAXP + 1];
    const int ngiven = nargs - nin - ax > 0 ? nargs - nin - ax : 0;
    if (param_values(b, args + nin + ax, ngiven < b->nparams ? ngiven : b->nparams, pv) == FAILURE) goto fail;
    int keepdims = 0;
    if (ax && b->keepdims_ok && nargs > nin + 1 + b->nparams) {
        zval *kd = &args[nin + 1 + b->nparams];
        ZVAL_DEREF(kd);
        keepdims = zend_is_true(kd);
    }
    int32_t ond[4];
    int64_t osh[4 * 32];
    int rc = tsr_fn_gufunc_shape(b->id, nin, in, masks, keepdims, pv, ond, osh);
    if (rc < 0) { throw_fn(rc, b->label); goto fail; }
    zval res[4];
    tsr_array outs[4];
    for (int o = 0; o < b->nout; o++) {
        int dt = (b->out_int >> o) & 1 ? D_I64 : ((b->out_bool >> o) & 1 ? D_BOOL : D_F64);
        const int d0 = in[0].dtype;
        if (((b->out_int_like >> o) & 1) && (d0 == D_I64 || d0 == D_I32 || d0 == D_U8)) dt = D_I64;
        if (o == 0 && b->keep_param >= 0 && (d0 == D_I64 || d0 == D_I32 || d0 == D_U8 || d0 == D_BOOL)
            && tsr_array_size(&in[0]) > 0) {             /* an empty sample: nanquantile's NaN is float64 */
            const double kv = pv[b->keep_param];
            if (kv >= 0 && kv < 32 && ((b->keep_mask >> (int)kv) & 1)) dt = d0;   /* numpy.quantile's discrete methods */
        }
        if (tsr_ndarray_new(&res[o], dt, ond[o], osh + o * 32, 0) == FAILURE) {
            for (int j = 0; j < o; j++) zval_ptr_dtor(&res[j]);
            goto fail;
        }
        outs[o] = Z_TSR(res[o])->a;
    }
    rc = tsr_fn_gufunc(b->id, nin, in, masks, keepdims, pv, b->nout, outs);
    for (int k = 0; k < nin; k++) zval_ptr_dtor(&arrs[k]);
    if (rc < 0) {
        for (int o = 0; o < b->nout; o++) zval_ptr_dtor(&res[o]);
        throw_fn(rc, b->label);
        return;
    }
    for (int o = 0; o < b->nout; o++) {
        const tsr_array *a = &Z_TSR(res[o])->a;
        if (a->ndim == 0) {
            zval s;
            const char *p0 = (const char *)a->data + a->offset;
            if (a->dtype == D_I64) ZVAL_LONG(&s, (zend_long)*(const int64_t *)p0);
            else if (a->dtype == D_I32) ZVAL_LONG(&s, (zend_long)*(const int32_t *)p0);
            else if (a->dtype == D_U8) ZVAL_LONG(&s, (zend_long)*(const uint8_t *)p0);
            else if (a->dtype == D_BOOL) ZVAL_BOOL(&s, *(const uint8_t *)p0 != 0);
            else if (a->dtype == D_F32) ZVAL_DOUBLE(&s, (double)*(const float *)p0);
            else ZVAL_DOUBLE(&s, *(const double *)p0);
            zval_ptr_dtor(&res[o]);
            ZVAL_COPY_VALUE(&res[o], &s);
        }
    }
    if (b->nout == 1) { ZVAL_COPY_VALUE(rv, &res[0]); return; }
    array_init_size(rv, (uint32_t)b->nout);
    for (int o = 0; o < b->nout; o++) add_assoc_zval(rv, b->outs[o], &res[o]);
    return;
fail:
    for (int k = 0; k < nin; k++) zval_ptr_dtor(&arrs[k]);
}

/* ================================================================ routine handler */

/* the widest routine signature in the registry is 17 arguments (12 samples and 5 options: levene, fligner) */
#define MAX_ROUTINE_ARGS 32

typedef struct { zval *z; int n, cap; } keep_list;

static void keep_free(keep_list *kl)
{
    for (int j = 0; j < kl->n; j++) zval_ptr_dtor(&kl->z[j]);
    if (kl->z) efree(kl->z);
    kl->z = NULL;
    kl->n = kl->cap = 0;
}

/* one PHP value -> one tsr_arg (arrays are converted and kept alive in kl) */
static int fill_arg(zval *z, tsr_arg *a, keep_list *kl)
{
    ZVAL_DEREF(z);
    switch (Z_TYPE_P(z)) {
    case IS_UNDEF: case IS_NULL: a->kind = 0; return SUCCESS;
    case IS_TRUE: case IS_FALSE: a->kind = 4; a->num = Z_TYPE_P(z) == IS_TRUE; return SUCCESS;
    case IS_LONG: a->kind = 1; a->num = (double)Z_LVAL_P(z); a->flags = 1; a->ival = (int64_t)Z_LVAL_P(z); return SUCCESS;
    case IS_DOUBLE: a->kind = 1; a->num = Z_DVAL_P(z); return SUCCESS;
    case IS_STRING: a->kind = 2; a->str = Z_STRVAL_P(z); return SUCCESS;
    default:
        if (kl->n == kl->cap) {
            kl->cap = kl->cap ? kl->cap * 2 : 8;
            kl->z = erealloc(kl->z, sizeof(zval) * (size_t)kl->cap);
        }
        if (tsr_ndarray_from_zval(z, -1, &kl->z[kl->n]) == FAILURE) return FAILURE;
        a->kind = 3;
        a->flags = Z_TYPE_P(z) == IS_ARRAY ? 2 : 0;            /* bit 1: built from a PHP list (a Python sequence) */
        a->arr = Z_TSR(kl->z[kl->n])->a;
        kl->n++;
        return SUCCESS;
    }
}

/* a sequence argument from a PHP list: kind 5 with one item per element (items are emalloc'ed into *blocks) */
static int fill_seq(zval *z, tsr_arg *a, keep_list *kl, tsr_arg **block)
{
    HashTable *ht = Z_ARRVAL_P(z);
    const uint32_t n = zend_hash_num_elements(ht);
    tsr_arg *items = ecalloc(n > 0 ? n : 1, sizeof(tsr_arg));
    *block = items;
    zval *e;
    uint32_t j = 0;
    ZEND_HASH_FOREACH_VAL(ht, e) {
        if (fill_arg(e, &items[j++], kl) == FAILURE) return FAILURE;
    } ZEND_HASH_FOREACH_END();
    a->kind = 5;
    a->items = items;
    a->count = n;
    return SUCCESS;
}

/* release a kernel result that will not be adopted (arrays, and sequences recursively) */
static void fn_results_free_ext(tsr_result *r)
{
    if (r->kind == 3 && r->arr.data) tsr_free(r->arr.data, r->bytes);
    if (r->kind == 6 && r->arr.data) {
        tsr_result *items = (tsr_result *)r->arr.data;
        for (int64_t j = 0; j < r->arr.shape[0]; j++) fn_results_free_ext(&items[j]);
        tsr_free(r->arr.data, r->bytes);
    }
    r->kind = 0;
}

/* a kernel result as a zval (kind 6 -> a PHP list); frees what it cannot adopt */
static int result_zval(tsr_result *r, zval *out)
{
    switch (r->kind) {
    case 1: ZVAL_DOUBLE(out, r->num); return SUCCESS;
    case 2: ZVAL_STRING(out, (const char *)r->arr.data); tsr_free(r->arr.data, r->bytes); return SUCCESS;  /* dtype name */
    case 4: ZVAL_BOOL(out, r->num != 0); return SUCCESS;
    case 5: ZVAL_LONG(out, (zend_long)r->ival); return SUCCESS;
    case 3:
        if (tsr_ndarray_adopt(out, r->arr.dtype, r->arr.ndim, r->arr.shape, r->arr.data, r->bytes) == FAILURE) {
            tsr_free(r->arr.data, r->bytes);
            ZVAL_UNDEF(out);
            return FAILURE;
        }
        return SUCCESS;
    case 6: {
        tsr_result *items = (tsr_result *)r->arr.data;
        const int64_t n = r->arr.shape[0];
        int ok = SUCCESS;
        array_init_size(out, (uint32_t)n);
        for (int64_t j = 0; j < n; j++) {
            if (ok == FAILURE) {                      /* after a failure: release the rest */
                if (items[j].kind == 3 || items[j].kind == 6) fn_results_free_ext(&items[j]);
                continue;
            }
            zval v;
            if (result_zval(&items[j], &v) == FAILURE) { ok = FAILURE; continue; }
            add_next_index_zval(out, &v);
        }
        tsr_free(r->arr.data, r->bytes);
        if (ok == FAILURE) { zval_ptr_dtor(out); ZVAL_UNDEF(out); }
        return ok;
    }
    default: ZVAL_UNDEF(out); return SUCCESS;
    }
}

/* a registry default as a PHP value */
static void default_zval(const char *d, zval *z)
{
    const size_t n = strlen(d);
    if (strcmp(d, "None") == 0) ZVAL_NULL(z);
    else if (strcmp(d, "True") == 0) ZVAL_TRUE(z);
    else if (strcmp(d, "False") == 0) ZVAL_FALSE(z);
    else if (strcmp(d, "inf") == 0) ZVAL_DOUBLE(z, INFINITY);
    else if (strcmp(d, "-inf") == 0) ZVAL_DOUBLE(z, -INFINITY);
    else if (n >= 2 && d[0] == '\'' && d[n - 1] == '\'') ZVAL_STRINGL(z, d + 1, n - 2);
    else if (strpbrk(d, ".eE")) ZVAL_DOUBLE(z, zend_strtod(d, NULL));
    else ZVAL_LONG(z, (zend_long)ZEND_STRTOL(d, NULL, 10));
}

static void call_routine(const fn_bind *b, zend_execute_data *execute_data, zval *args, int nargs, zval *rv)
{
    tsr_arg cargs[MAX_ROUTINE_ARGS];
    tsr_arg *blocks[MAX_ROUTINE_ARGS];
    int nblocks = 0;
    zval defs[MAX_ROUTINE_ARGS];
    int ndefs = 0;
    keep_list kl = {NULL, 0, 0};
    memset(cargs, 0, sizeof cargs);
    int ncall = nargs;
    if (b->variadic >= 0) {
        /* numpy's f(a, *xi, option=...): positional arguments up to the variadic one, the rest form the
           sequence; options come as named arguments (collected by the engine as extra named parameters) */
        HashTable *named = (ZEND_CALL_INFO(execute_data) & ZEND_CALL_HAS_EXTRA_NAMED_PARAMS) ? execute_data->extra_named_params : NULL;
        const int var = b->variadic;
        if (nargs < var) { zend_argument_count_error("%s() expects at least %d arguments, %d given", b->label, var, nargs); return; }
        ncall = b->nrargs;
        for (int k = 0; k < var; k++) if (fill_arg(&args[k], &cargs[k], &kl) == FAILURE) goto done;
        const int nv = nargs - var;
        tsr_arg *items = ecalloc(nv > 0 ? nv : 1, sizeof(tsr_arg));
        blocks[nblocks++] = items;
        for (int j = 0; j < nv; j++) if (fill_arg(&args[var + j], &items[j], &kl) == FAILURE) goto done;
        cargs[var].kind = 5;
        cargs[var].items = items;
        cargs[var].count = nv;
        int used = 0;
        for (int k = var + 1; k < b->nrargs; k++) {
            zval *z = named ? zend_hash_str_find(named, b->rnames[k], strlen(b->rnames[k])) : NULL;
            if (z) used++;
            else { default_zval(b->rdefs[k] ? b->rdefs[k] : "None", &defs[ndefs]); z = &defs[ndefs++]; }
            if (fill_arg(z, &cargs[k], &kl) == FAILURE) goto done;
        }
        if (named && (int)zend_hash_num_elements(named) > used) {
            zend_string *key;
            ZEND_HASH_FOREACH_STR_KEY(named, key) {
                int known = 0;
                for (int k = var + 1; k < b->nrargs; k++) if (key && zend_string_equals_cstr(key, b->rnames[k], strlen(b->rnames[k]))) known = 1;
                if (!known) { zend_throw_error(NULL, "Unknown named parameter $%s", key ? ZSTR_VAL(key) : "?"); goto done; }
            } ZEND_HASH_FOREACH_END();
        }
    } else {
        if (nargs > MAX_ROUTINE_ARGS) { zend_argument_count_error("%s(): too many arguments", b->label); return; }
        /* omitted trailing arguments get their registry defaults, as the FFI façades pass them */
        if (b->nrargs > nargs && b->nrargs <= MAX_ROUTINE_ARGS) ncall = b->nrargs;
        for (int k = 0; k < ncall; k++) {
            zval *z;
            if (k < nargs) z = &args[k];
            else { default_zval(b->rdefs[k] ? b->rdefs[k] : "None", &defs[ndefs]); z = &defs[ndefs++]; }
            ZVAL_DEREF(z);
            if ((b->inplace >> k) & 1) {
                if (!is_ndarray(z)) {
                    zend_type_error("%s(): argument #%d must be a Tessero\\Ext\\NDArray (it is modified in place), %s given", b->label, k + 1, zend_zval_type_name(z));
                    goto done;
                }
                if (tsr_root(Z_TSR_P(z))->flags & TSR_F_READONLY) { zend_value_error("%s(): assignment destination is read-only", b->label); goto done; }
            }
            if (((b->seq >> k) & 1) && Z_TYPE_P(z) == IS_ARRAY) {
                if (fill_seq(z, &cargs[k], &kl, &blocks[nblocks++]) == FAILURE) goto done;
            } else if (fill_arg(z, &cargs[k], &kl) == FAILURE) goto done;
        }
    }
    {
        tsr_result res[16];
        const int nres = b->nout > 0 ? (b->nout < 16 ? b->nout : 16) : 1;
        const int rc = tsr_fn_routine(b->id, ncall, cargs, nres, res);
        keep_free(&kl);
        if (rc < 0) { throw_fn(rc, b->label); goto done; }
        zval vals[16];
        int present = 0, last = -1, failed = 0;
        for (int k = 0; k < nres; k++) {
            ZVAL_UNDEF(&vals[k]);
            if (failed) { if (res[k].kind == 3 || res[k].kind == 6) fn_results_free_ext(&res[k]); continue; }
            if (res[k].kind == 0) continue;
            if (result_zval(&res[k], &vals[k]) == FAILURE) { failed = 1; continue; }
            present++;
            last = k;
        }
        if (failed) {
            for (int k = 0; k < nres; k++) if (Z_TYPE(vals[k]) != IS_UNDEF) zval_ptr_dtor(&vals[k]);
            goto done;
        }
        if (present == 0 && b->nout == 0) { ZVAL_NULL(rv); goto done; }   /* in-place writers return None */
        if (present == 1) { ZVAL_COPY_VALUE(rv, &vals[last]); goto done; }
        array_init_size(rv, (uint32_t)present);
        for (int k = 0; k < nres; k++) if (Z_TYPE(vals[k]) != IS_UNDEF) add_assoc_zval(rv, b->outs[k], &vals[k]);
    }
done:
    keep_free(&kl);
    for (int j = 0; j < nblocks; j++) efree(blocks[j]);
    for (int j = 0; j < ndefs; j++) zval_ptr_dtor(&defs[j]);
}

/* ================================================================ dispatch */

static const fn_bind *bind_for(zend_execute_data *execute_data)
{
    zend_function *f = EX(func);
    char key[160];
    snprintf(key, sizeof key, "%s::%s", ZSTR_VAL(f->common.scope->name), ZSTR_VAL(f->common.function_name));
    for (char *p = key; *p; p++) *p = (char)tolower((unsigned char)*p);
    zval *v = zend_hash_str_find(&g_by_method, key, strlen(key));
    return v ? &g_binds[Z_LVAL_P(v)] : NULL;
}

uint64_t *tsr_generator_state(zval *obj);

/* a Generator method (kind "random"): arrays..., size, enum params...; NumPy's stream exactly */
static void call_random(const fn_bind *b, zval *self, zval *args, int nargs, zval *rv)
{
    const int nin = b->nin;
    zval ops[8];
    int nops = 0;
    for (int k = 0; k < nin; k++) {
        zval *z = k < nargs ? &args[k] : NULL;
        if (z) ZVAL_DEREF(z);
        if (!z || Z_TYPE_P(z) == IS_UNDEF) { zend_argument_count_error("%s() needs argument %d", b->label, k + 1); goto fail; }
        if (is_ndarray(z) && Z_TSR_P(z)->a.dtype == D_C128) {
            zend_throw_exception_ex(tsr_ce_dtype_exception, 0, "%s() does not support complex128 input", b->label);
            goto fail;
        }
        zval tmp;
        if (tsr_ndarray_from_zval(z, is_ndarray(z) ? -1 : D_F64, &tmp) == FAILURE) goto fail;
        if (tsr_ndarray_contiguous_as(&tmp, D_F64, &ops[nops]) == FAILURE) { zval_ptr_dtor(&tmp); goto fail; }
        zval_ptr_dtor(&tmp);
        nops++;
    }
    int32_t bnd = 0;
    int64_t bshape[32];
    for (int k = 0; k < nops; k++) {
        const tsr_array *a = &Z_TSR(ops[k])->a;
        if (k == 0) { bnd = a->ndim; memcpy(bshape, a->shape, sizeof(int64_t) * (size_t)a->ndim); continue; }
        int32_t rnd;
        int64_t rsh[32];
        if (tsr_broadcast_shape(bnd, bshape, a->ndim, a->shape, &rnd, rsh) != 0) {
            zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s(): shape mismatch: objects cannot be broadcast to a single shape", b->label);
            goto fail;
        }
        bnd = rnd;
        memcpy(bshape, rsh, sizeof(int64_t) * (size_t)bnd);
    }
    zval *size = nargs > nin ? &args[nin] : NULL;
    if (size) ZVAL_DEREF(size);
    int32_t nd;
    int64_t shape[32];
    const int size_null = size == NULL || Z_TYPE_P(size) == IS_NULL || Z_TYPE_P(size) == IS_UNDEF;
    if (size_null) {
        nd = bnd;
        memcpy(shape, bshape, sizeof(int64_t) * (size_t)bnd);
    } else {
        nd = 0;
        if (Z_TYPE_P(size) == IS_LONG) shape[nd++] = Z_LVAL_P(size);
        else if (Z_TYPE_P(size) == IS_ARRAY && zend_hash_num_elements(Z_ARRVAL_P(size)) <= 32) {
            zval *e;
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(size), e) {
                if (Z_TYPE_P(e) != IS_LONG) { zend_type_error("%s(): size must hold integers", b->label); goto fail; }
                shape[nd++] = Z_LVAL_P(e);
            } ZEND_HASH_FOREACH_END();
        } else { zend_type_error("%s(): size must be an int, a list of ints or null", b->label); goto fail; }
        for (int d = 0; d < nd; d++) if (shape[d] < 0) { zend_value_error("%s(): negative dimensions are not allowed", b->label); goto fail; }
        int32_t rnd;
        int64_t rsh[32];
        if (tsr_broadcast_shape(bnd, bshape, nd, shape, &rnd, rsh) != 0 || rnd != nd || memcmp(rsh, shape, sizeof(int64_t) * (size_t)nd) != 0) {
            zend_value_error("%s(): size is not compatible with the parameters' broadcast shape", b->label);
            goto fail;
        }
    }
    double pv[MAXP + 1];
    const int ngiven = nargs - nin - 1 > 0 ? nargs - nin - 1 : 0;
    if (param_values(b, args + nin + 1, ngiven < b->nparams ? ngiven : b->nparams, pv) == FAILURE) goto fail;
    int f32 = 0;
    for (int p = 0; p < b->nparams; p++) if (strcmp(b->params[p].name, "dtype") == 0 && pv[p] == 1.0) f32 = 1;
    const int dt = b->out_int ? D_I64 : (f32 ? D_F32 : D_F64);
    zval res;
    if (tsr_ndarray_new(&res, dt, nd, shape, 0) == FAILURE) goto fail;
    void *ptrs[9];
    int64_t strides[9 * 32];
    for (int k = 0; k < nops; k++) {
        const tsr_array *a = &Z_TSR(ops[k])->a;
        ptrs[k] = (char *)a->data + a->offset;
        if (nd > 0) tsr_broadcast_strides(a, nd, shape, strides + (size_t)k * nd);
    }
    const tsr_array *o = &Z_TSR(res)->a;
    ptrs[nops] = (char *)o->data + o->offset;
    for (int d = 0; d < nd; d++) strides[(size_t)nops * nd + d] = o->strides[d];
    const int rc = tsr_fn_random(b->id, tsr_generator_state(self), pv, nd, nd ? shape : NULL, nops + 1, ptrs, nd ? strides : NULL, dt);
    for (int k = 0; k < nops; k++) zval_ptr_dtor(&ops[k]);
    if (rc < 0) { zval_ptr_dtor(&res); throw_fn(rc, b->label); return; }
    if (size_null && nd == 0) {
        if (dt == D_I64) ZVAL_LONG(rv, (zend_long)*(int64_t *)ptrs[nops]);
        else if (dt == D_F32) ZVAL_DOUBLE(rv, (double)*(float *)ptrs[nops]);
        else ZVAL_DOUBLE(rv, *(double *)ptrs[nops]);
        zval_ptr_dtor(&res);
        return;
    }
    ZVAL_COPY_VALUE(rv, &res);
    return;
fail:
    for (int k = 0; k < nops; k++) zval_ptr_dtor(&ops[k]);
}

static PHP_METHOD(Fn, call)
{
    const fn_bind *b = bind_for(execute_data);
    if (!b) { zend_throw_error(NULL, "unknown Tessero function"); RETURN_THROWS(); }
    const int nargs = (int)ZEND_NUM_ARGS();
    zval *args = ZEND_CALL_ARG(execute_data, 1);
    if (b->kind == K_UFUNC) {
        /* the last parameter is out */
        zval *out = nargs > b->nin ? &args[b->nin] : NULL;
        call_ufunc(b, 0, args, nargs > b->nin ? b->nin : nargs, out, return_value);
    } else if (b->kind == K_GUFUNC) {
        call_gufunc(b, args, nargs, return_value);
    } else if (b->kind == K_ROUTINE) {
        call_routine(b, execute_data, args, nargs, return_value);
    } else if (b->kind == K_RANDOM) {
        call_random(b, ZEND_THIS, args, nargs, return_value);
    } else {
        /* distribution factory: Stats::norm(loc, scale) -> Tessero\Ext\Distribution */
        tsr_distribution_new(return_value, b->id, b->label, args, nargs);
    }
    if (EG(exception)) RETURN_THROWS();
}

/* Tessero\Ext\Distribution methods call this: evaluate distribution `id`, method `m` on (x?, params) */
void tsr_fn_dist_call(int id, int method, zval *args, int nargs, zval *rv)
{
    if (id >= 0 && id < g_nbinds && g_binds[id].kind == K_DIST) { call_ufunc(&g_binds[id], method, args, nargs, NULL, rv); return; }
    zend_throw_error(NULL, "unknown distribution");
}

/* Invoke a registry routine by name (for NDArray methods that delegate to np.* kernels, e.g. round/take/put).
   Only non-variadic routines; execute_data is unused on that path, so NULL is safe. */
void tsr_fn_call_routine(const char *name, zval *args, int nargs, zval *rv)
{
    const int id = tsr_fn_find(name);
    if (id < 0 || id >= g_nbinds || g_binds[id].kind != K_ROUTINE) { zend_throw_error(NULL, "unknown Tessero routine '%s'", name); return; }
    call_routine(&g_binds[id], NULL, args, nargs, rv);
}


/* ================================================================ registration */

static char *pstrdup(const char *s) { return pestrdup(s, 1); }

static void load_bind(int id, zval *info, fn_bind *b)
{
    memset(b, 0, sizeof *b);
    b->id = id;
    zval *v;
    const char *kind = (v = zend_hash_str_find(Z_ARRVAL_P(info), "kind", 4)) ? Z_STRVAL_P(v) : "";
    b->kind = strcmp(kind, "ufunc") == 0 ? K_UFUNC : strcmp(kind, "gufunc") == 0 ? K_GUFUNC : strcmp(kind, "routine") == 0 ? K_ROUTINE
            : strcmp(kind, "random") == 0 ? K_RANDOM : K_DIST;
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "nin", 3))) b->nin = (int)Z_LVAL_P(v);
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "nout", 4))) b->nout = (int)Z_LVAL_P(v);
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "outs", 4)) && Z_TYPE_P(v) == IS_ARRAY) {
        zval *e;
        int k = 0;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(v), e) { if (k < 16) snprintf(b->outs[k++], 24, "%s", Z_STRVAL_P(e)); } ZEND_HASH_FOREACH_END();
    }
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "int_args", 8)) && Z_TYPE_P(v) == IS_ARRAY) {
        zval *e;
        b->has_int = 1;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(v), e) { b->int_args |= 1u << Z_LVAL_P(e); } ZEND_HASH_FOREACH_END();
    }
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "params", 6)) && Z_TYPE_P(v) == IS_ARRAY) {
        zval *p;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(v), p) {
            if (b->nparams >= MAXP) break;
            fn_param *fp = &b->params[b->nparams++];
            zval *f;
            if ((f = zend_hash_str_find(Z_ARRVAL_P(p), "name", 4))) snprintf(fp->name, sizeof fp->name, "%s", Z_STRVAL_P(f));
            if ((f = zend_hash_str_find(Z_ARRVAL_P(p), "enum", 4)) && Z_TYPE_P(f) == IS_ARRAY) {
                zval *e;
                fp->is_enum = 1;
                ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(f), e) { if (fp->nenum < MAXE) snprintf(fp->enums[fp->nenum++], sizeof fp->enums[0], "%s", Z_STRVAL_P(e)); } ZEND_HASH_FOREACH_END();
            } else if ((f = zend_hash_str_find(Z_ARRVAL_P(p), "default", 7))) {
                const char *d = Z_STRVAL_P(f);
                fp->is_bool = strcmp(d, "True") == 0 || strcmp(d, "False") == 0;
                fp->def = strcmp(d, "None") == 0 ? NAN : strcmp(d, "True") == 0 ? 1.0 : strcmp(d, "False") == 0 ? 0.0
                        : strcmp(d, "inf") == 0 ? INFINITY : strcmp(d, "-inf") == 0 ? -INFINITY : zend_strtod(d, NULL);
            } else {
                fp->def = NAN;
            }
        } ZEND_HASH_FOREACH_END();
    }
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "axis", 4))) {
        b->axis_hidden = strcmp(Z_STRVAL_P(v), "hidden") == 0;
        b->axis_none = b->axis_hidden || strcmp(Z_STRVAL_P(v), "None") == 0;
        b->axis_default = b->axis_none ? 0 : atoi(Z_STRVAL_P(v));
    }
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "axis_inputs", 11))) b->axis_inputs = (int)Z_LVAL_P(v);
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "out_int", 7))) b->out_int = (int)Z_LVAL_P(v);
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "out_bool", 8))) b->out_bool = (int)Z_LVAL_P(v);
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "out_int_like", 12))) b->out_int_like = (int)Z_LVAL_P(v);
    b->keep_param = -1;
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "keep_param", 10))) b->keep_param = (int)Z_LVAL_P(v);
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "keep_mask", 9))) b->keep_mask = (int)Z_LVAL_P(v);
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "axis_single", 11))) b->axis_single = (int)Z_LVAL_P(v);
    b->keepdims_ok = 1;
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "keepdims", 8))) b->keepdims_ok = (int)Z_LVAL_P(v);
    if (b->kind == K_ROUTINE && (v = zend_hash_str_find(Z_ARRVAL_P(info), "args", 4)) && Z_TYPE_P(v) == IS_ARRAY) {
        b->nin = zend_hash_num_elements(Z_ARRVAL_P(v));
        zval *defaults = zend_hash_str_find(Z_ARRVAL_P(info), "defaults", 8), *e;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(v), e) {
            if (b->nrargs >= 32) break;
            char t[64];
            camel(Z_STRVAL_P(e), t, sizeof t);
            b->rnames[b->nrargs] = pestrdup(t, 1);
            zval *d = defaults && Z_TYPE_P(defaults) == IS_ARRAY ? zend_hash_find(Z_ARRVAL_P(defaults), Z_STR_P(e)) : NULL;
            b->rdefs[b->nrargs] = d && Z_TYPE_P(d) == IS_STRING ? pestrdup(Z_STRVAL_P(d), 1) : NULL;
            b->nrargs++;
        } ZEND_HASH_FOREACH_END();
    }
    b->variadic = -1;
    if (b->kind == K_ROUTINE) {
        if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "variadic", 8))) b->variadic = (int)Z_LVAL_P(v);
        if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "out_seq", 7))) b->out_seq = (int)Z_LVAL_P(v);
        if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "inplace", 7)) && Z_TYPE_P(v) == IS_ARRAY) {
            zval *e;
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(v), e) { if (Z_LVAL_P(e) < 32) b->inplace |= 1u << Z_LVAL_P(e); } ZEND_HASH_FOREACH_END();
        }
        if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "seq", 3)) && Z_TYPE_P(v) == IS_ARRAY) {
            zval *e;
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(v), e) { if (Z_LVAL_P(e) < 32) b->seq |= 1u << Z_LVAL_P(e); } ZEND_HASH_FOREACH_END();
        }
    }
    if (b->kind == K_GUFUNC && (v = zend_hash_str_find(Z_ARRVAL_P(info), "defaults", 8)) && Z_TYPE_P(v) == IS_ARRAY) {
        zval *args = zend_hash_str_find(Z_ARRVAL_P(info), "args", 4), *e;
        int k = 0;
        if (args && Z_TYPE_P(args) == IS_ARRAY) {
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(args), e) {
                zval *d = zend_hash_find(Z_ARRVAL_P(v), Z_STR_P(e));
                if (d && k < 4) {
                    b->optional |= 1u << k;
                    char *end = NULL;
                    const double x = Z_TYPE_P(d) == IS_STRING ? strtod(Z_STRVAL_P(d), &end) : NAN;
                    b->in_def[k] = (end && end != Z_STRVAL_P(d) && *end == '\0') ? x : NAN;
                }
                k++;
            } ZEND_HASH_FOREACH_END();
        }
    }
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "discrete", 8))) b->discrete = Z_TYPE_P(v) == IS_TRUE;
    if ((v = zend_hash_str_find(Z_ARRVAL_P(info), "shapes", 6)) && Z_TYPE_P(v) == IS_ARRAY) b->nshape = zend_hash_num_elements(Z_ARRVAL_P(v));
}

static char *camel_dup(const char *s) { char t[64]; camel(s, t, sizeof t); return pestrdup(t, 1); }

/* a PHP literal for a registry default ("None", "False", "10", "'valid'") */
static char *php_default(const char *d)
{
    if (strcmp(d, "None") == 0) return (char *)"null";
    if (strcmp(d, "True") == 0) return (char *)"true";
    if (strcmp(d, "False") == 0) return (char *)"false";
    if (strcmp(d, "inf") == 0) return (char *)"INF";
    if (strcmp(d, "-inf") == 0) return (char *)"-INF";
    return pstrdup(d);
}

/* build arginfo for one function: [0] = return/required count, then one entry per parameter */
static zend_internal_arg_info *make_arginfo(const fn_bind *b, zval *info, int *nargs_out)
{
    int total = 0, required = 0;
    zend_internal_arg_info *ai;
    zval *args = zend_hash_str_find(Z_ARRVAL_P(info), "args", 4);
    zval *defaults = zend_hash_str_find(Z_ARRVAL_P(info), "defaults", 8);
    zval *shapes = zend_hash_str_find(Z_ARRVAL_P(info), "shapes", 6);
    if (b->kind == K_UFUNC) {
        total = b->nin + 1;
        required = b->nin;
    } else if (b->kind == K_GUFUNC) {
        total = b->nin + 1 + b->nparams + 1;
        required = b->nin;
    } else if (b->kind == K_ROUTINE) {
        total = b->nin;
        required = 0;
    } else if (b->kind == K_RANDOM) {
        total = b->nin + 1 + b->nparams;
        required = 0;
    } else {
        total = b->nshape + (b->discrete ? 1 : 2);
        required = b->nshape;
    }
    ai = pecalloc((size_t)total + 1, sizeof(zend_internal_arg_info), 1);
    ai[0].type = (zend_type)ZEND_TYPE_INIT_NONE(0);
    int k = 1;
    zval *e;
    if (b->kind == K_DIST) {
        ai[0].name = (const char *)(uintptr_t)required;
        if (shapes && Z_TYPE_P(shapes) == IS_ARRAY) {
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shapes), e) {
                ai[k].name = camel_dup(Z_STRVAL_P(e));
                ai[k].type = (zend_type)ZEND_TYPE_INIT_NONE(0);
                k++;
            } ZEND_HASH_FOREACH_END();
        }
        ai[k].name = "loc"; ai[k].type = (zend_type)ZEND_TYPE_INIT_NONE(0); ai[k].default_value = "0"; k++;
        if (!b->discrete) { ai[k].name = "scale"; ai[k].type = (zend_type)ZEND_TYPE_INIT_NONE(0); ai[k].default_value = "1"; k++; }
        *nargs_out = k - 1;
        return ai;
    }
    int nreq = 0, seen_default = 0;
    if (b->kind == K_ROUTINE && b->variadic >= 0) {
        /* f(a, ...$xi): the arguments before the variadic one, then the variadic; options arrive as extra
           named parameters (ZEND_ACC_VARIADIC is set from the last arg_info) */
        for (int i = 0; i <= b->variadic; i++) {
            ai[k].name = b->rnames[i];
            ai[k].type = (zend_type)ZEND_TYPE_INIT_NONE(i == b->variadic ? _ZEND_ARG_INFO_FLAGS(0, 1, 0) : 0);
            k++;
        }
        ai[0].name = (const char *)(uintptr_t)b->variadic;
        *nargs_out = k - 1;
        return ai;
    }
    if (args && Z_TYPE_P(args) == IS_ARRAY) {
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(args), e) {
            ai[k].name = camel_dup(Z_STRVAL_P(e));
            ai[k].type = (zend_type)ZEND_TYPE_INIT_NONE(0);
            zval *d = defaults && Z_TYPE_P(defaults) == IS_ARRAY ? zend_hash_find(Z_ARRVAL_P(defaults), Z_STR_P(e)) : NULL;
            if (d) { ai[k].default_value = php_default(Z_STRVAL_P(d)); seen_default = 1; }
            else if (!seen_default) nreq++;
            k++;
        } ZEND_HASH_FOREACH_END();
    }
    required = b->kind == K_UFUNC ? b->nin : nreq;
    if (b->kind == K_RANDOM) {
        ai[k].name = "size"; ai[k].type = (zend_type)ZEND_TYPE_INIT_NONE(0); ai[k].default_value = "null"; k++;
        for (int p = 0; p < b->nparams; p++) {
            ai[k].name = camel_dup(b->params[p].name);
            ai[k].type = (zend_type)ZEND_TYPE_INIT_NONE(0);
            char t[64];
            snprintf(t, sizeof t, "'%s'", b->params[p].enums[0]);
            ai[k].default_value = pstrdup(t);
            k++;
        }
        ai[0].name = (const char *)(uintptr_t)required;
        *nargs_out = k - 1;
        return ai;
    }
    ai[0].name = (const char *)(uintptr_t)required;
    if (b->kind == K_UFUNC) {
        ai[k].name = "out";
        ai[k].type = (zend_type)ZEND_TYPE_INIT_NONE(0);
        ai[k].default_value = "null";
        k++;
    } else if (b->kind == K_GUFUNC) {
        if (!b->axis_hidden) {
            ai[k].name = "axis";
            ai[k].type = (zend_type)ZEND_TYPE_INIT_NONE(0);
            if (b->axis_none) ai[k].default_value = "null";
            else { char t[16]; snprintf(t, sizeof t, "%d", b->axis_default); ai[k].default_value = pstrdup(t); }
            k++;
        }
        for (int p = 0; p < b->nparams; p++) {
            const fn_param *fp = &b->params[p];
            ai[k].name = camel_dup(fp->name);
            ai[k].type = (zend_type)ZEND_TYPE_INIT_NONE(0);
            char t[64];
            if (fp->is_enum) snprintf(t, sizeof t, "'%s'", fp->enums[0]);
            else if (fp->is_bool) snprintf(t, sizeof t, "%s", fp->def != 0.0 ? "true" : "false");
            else if (isnan(fp->def)) snprintf(t, sizeof t, "null");
            else if (isinf(fp->def)) snprintf(t, sizeof t, fp->def > 0 ? "INF" : "-INF");
            else if (fp->def == floor(fp->def) && fabs(fp->def) < 1e15) snprintf(t, sizeof t, "%lld", (long long)fp->def);
            else snprintf(t, sizeof t, "%.17g", fp->def);
            ai[k].default_value = pstrdup(t);
            k++;
        }
        if (!b->axis_hidden && b->keepdims_ok) {
            ai[k].name = "keepdims";
            ai[k].type = (zend_type)ZEND_TYPE_INIT_NONE(0);
            ai[k].default_value = "false";
            k++;
        }
    }
    *nargs_out = k - 1;
    return ai;
}

void tsr_register_fn(void)
{
    zend_hash_init(&g_by_method, 512, NULL, NULL, 1);
    const int n = tsr_fn_count();
    g_binds = pecalloc((size_t)(n > 0 ? n : 1), sizeof(fn_bind), 1);
    g_nbinds = n;
    /* module index 3 (random) goes to Tessero\Ext\Random\Generator (instance methods, tessero_random.c); the
       others are abstract classes of static methods. linalg (index 4) is Tessero\Ext\Linalg. */
    static const char *const MODS[12][2] = {{"special", "Special"}, {"stats", "Stats"}, {"np", "Np"}, {"random", "Random\\Generator"}, {"linalg", "Linalg"}, {"slinalg", "ScipyLinalg"}, {"distance", "Distance"}, {"sparse", "Sparse"}, {"signal", "Signal"}, {"ndimage", "Ndimage"}, {"csgraph", "Csgraph"}, {"windows", "SignalWindows"}};
    zend_function_entry *fes[12];
    int nfe[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int m = 0; m < 12; m++) fes[m] = pecalloc((size_t)n + 1, sizeof(zend_function_entry), 1);
    char buf[16384];
    for (int id = 0; id < n; id++) {
        const int64_t len = tsr_fn_info(id, buf, sizeof buf);
        if (len <= 0 || len >= (int64_t)sizeof buf) continue;
        zval info;
        if (php_json_decode_ex(&info, buf, (size_t)len, PHP_JSON_OBJECT_AS_ARRAY, 16) == FAILURE || Z_TYPE(info) != IS_ARRAY) {
            zval_ptr_dtor(&info);
            continue;
        }
        fn_bind *b = &g_binds[id];
        load_bind(id, &info, b);
        const char *full = tsr_fn_name(id);
        const char *dot = strchr(full, '.');
        int m = -1;
        for (int k = 0; k < 12; k++)
            if (dot && (size_t)(dot - full) == strlen(MODS[k][0]) && strncmp(full, MODS[k][0], (size_t)(dot - full)) == 0) m = k;
        if (m < 0) { zval_ptr_dtor(&info); continue; }
        char method[64];
        camel(dot + 1, method, sizeof method);
        snprintf(b->label, sizeof b->label, "%s::%s", m == 3 ? "Generator" : MODS[m][1], method);
        int nargs = 0;
        zend_internal_arg_info *ai = make_arginfo(b, &info, &nargs);
        zval_ptr_dtor(&info);
        zend_function_entry *fe = &fes[m][nfe[m]++];
        fe->fname = pstrdup(method);
        fe->handler = zim_Fn_call;
        fe->arg_info = ai;
        fe->num_args = (uint32_t)nargs;
        fe->flags = m == 3 ? ZEND_ACC_PUBLIC : ZEND_ACC_PUBLIC | ZEND_ACC_STATIC;
        char key[160];
        snprintf(key, sizeof key, "tessero\\ext\\%s::%s", MODS[m][1], method);
        for (char *p = key; *p; p++) *p = (char)tolower((unsigned char)*p);
        zval idx;
        ZVAL_LONG(&idx, id);
        zend_hash_str_update(&g_by_method, key, strlen(key), &idx);
    }
    tsr_register_generator(fes[3], nfe[3]);
    tsr_stats_register_pb(fes[1], &nfe[1]);   /* poisson_binom: a vector-shape distribution, hand-written on Stats */
    /* the abstract static-method classes: special/stats/np (indices 0-2) and linalg (index 4) */
    struct { zend_class_entry **ce; int m; } statics[] = {
        {&tsr_ce_special, 0}, {&tsr_ce_stats, 1}, {&tsr_ce_np, 2}, {&tsr_ce_linalg, 4}, {&tsr_ce_slinalg, 5}, {&tsr_ce_distance, 6}, {&tsr_ce_sparse, 7}, {&tsr_ce_signal, 8}, {&tsr_ce_ndimage, 9}, {&tsr_ce_csgraph, 10}, {&tsr_ce_signal_windows, 11}};
    for (int j = 0; j < 11; j++) {
        const int m = statics[j].m;
        zend_class_entry ce;
        INIT_CLASS_ENTRY_EX(ce, "", 0, fes[m]);
        char cname[64];
        snprintf(cname, sizeof cname, "Tessero\\Ext\\%s", MODS[m][1]);
        ce.name = zend_string_init_interned(cname, strlen(cname), 1);
        *statics[j].ce = zend_register_internal_class(&ce);
        (*statics[j].ce)->ce_flags |= ZEND_ACC_EXPLICIT_ABSTRACT_CLASS | ZEND_ACC_NO_DYNAMIC_PROPERTIES;
    }
}
