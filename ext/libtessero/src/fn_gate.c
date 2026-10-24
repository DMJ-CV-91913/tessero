/*
 * The registry's input gate (ADR 0016): structural checks of every caller-supplied argument, run by each
 * tsr_fn_* entry point before any routine, loop or core function sees the data.
 *
 * What it guarantees once a call passes: every tsr_arg has a defined kind and only defined flag bits; integer
 * numbers agree with their double; strings are non-NULL and NUL-terminated within FN_GATE_MAXSTR bytes; every
 * array has a valid dtype, 0 <= ndim <= 32, non-negative dimensions whose element count and byte extent fit in
 * int64 without overflow, a non-NULL data pointer when it has elements, element-aligned addresses, and every
 * element at or after the start of its block (data + offset + the most negative stride reach >= data), and, for
 * outputs, no zero stride on an axis longer than 1;
 * sequences are bounded in length and depth; strided operand sets (ufunc, random, rvs) have non-negative
 * dimensions, aligned operands and steps, stride extents that do not overflow, and outputs whose elements do not
 * alias through a zero stride. What stays each routine's job: the meaning of the values
 * (domains, integrality of float parameters, shape agreement between arguments, NaN policy). The checks are
 * O(nargs + ndim) per call and never read array elements.
 */
#include "fn.h"

#include <stdint.h>
#include <string.h>

#define FN_GATE_MAXSTR ((int64_t)INT32_MAX)        /* routines hold string lengths in int */
#define FN_GATE_MAXSEQ ((int64_t)1 << 24)          /* items in one sequence argument */
#define FN_GATE_MAXDEPTH 4                         /* nesting of sequence arguments */

static int64_t gate_elsize(int32_t dtype)
{
    switch (dtype) {
    case TSR_F64: case TSR_I64: return 8;
    case TSR_F32: case TSR_I32: return 4;
    case TSR_U8: case TSR_BOOL: return 1;
    case TSR_C128: return 16;
    default: return 0;
    }
}

/* a += b and a *= b on int64, 0 on overflow */
static int gate_add(int64_t *a, int64_t b)
{
    if ((b > 0 && *a > INT64_MAX - b) || (b < 0 && *a < INT64_MIN - b)) return 0;
    *a += b;
    return 1;
}

static int gate_mul(int64_t a, int64_t b, int64_t *out)
{
    if (a == 0 || b == 0) { *out = 0; return 1; }
    if (a == INT64_MIN || b == INT64_MIN) return 0;
    const int64_t ua = a < 0 ? -a : a, ub = b < 0 ? -b : b;
    if (ua > INT64_MAX / ub) return 0;
    *out = a * b;
    return 1;
}

/* dimensions and element count: TSR_OK, or the code with the message set */
static int gate_shape(int32_t ndim, const int64_t *shape, const char *what, int k, int64_t *size)
{
    if (ndim < 0) { fn_set_error("%s %d: ndim %d is negative", what, k, (int)ndim); return TSR_EARG; }
    if (ndim > TSR_MAXDIM) { fn_set_error("%s %d: ndim %d exceeds the maximum of %d", what, k, (int)ndim, TSR_MAXDIM); return TSR_EDIM; }
    if (ndim > 0 && !shape) { fn_set_error("%s %d: shape is NULL", what, k); return TSR_EARG; }
    for (int32_t d = 0; d < ndim; d++)
        if (shape[d] < 0) { fn_set_error("%s %d: negative dimension %lld on axis %d", what, k, (long long)shape[d], (int)d); return TSR_EARG; }
    const int64_t n = tsr_shape_size(ndim, shape);
    if (n < 0) { fn_set_error("%s %d: the element count overflows", what, k); return TSR_EARG; }
    *size = n;
    return TSR_OK;
}

/* the byte span [lo, hi) the elements of one strided operand cover relative to its first element */
static int gate_span(int32_t ndim, const int64_t *shape, const int64_t *strides, int64_t elsize, int64_t *lo, int64_t *hi)
{
    int64_t l = 0, h = 0;
    for (int32_t d = 0; d < ndim; d++) {
        if (shape[d] <= 1) continue;
        int64_t reach;
        if (!gate_mul(shape[d] - 1, strides[d], &reach)) return 0;
        if (!gate_add(reach < 0 ? &l : &h, reach)) return 0;
    }
    if (!gate_add(&h, elsize)) return 0;
    *lo = l;
    *hi = h;
    return 1;
}

int fn_gate_array(const tsr_array *a, int with_data, const char *what, int k)
{
    if (!a) { fn_set_error("%s %d: array is NULL", what, k); return TSR_EARG; }
    const int64_t el = gate_elsize(a->dtype);
    if (el == 0) { fn_set_error("%s %d: unknown dtype %d", what, k, (int)a->dtype); return TSR_ETYPE; }
    int64_t size;
    const int rc = gate_shape(a->ndim, a->shape, what, k, &size);
    if (rc != TSR_OK || !with_data || size == 0) return rc;
    if (!a->data) { fn_set_error("%s %d: data is NULL for %lld elements", what, k, (long long)size); return TSR_EARG; }
    int64_t lo, hi;
    if (!gate_span(a->ndim, a->shape, a->strides, el, &lo, &hi)
        || !gate_add(&lo, a->offset) || !gate_add(&hi, a->offset)) {
        fn_set_error("%s %d: strides and offset overflow the address range", what, k);
        return TSR_EARG;
    }
    if (lo < 0) { fn_set_error("%s %d: an element lies %lld bytes before the start of its block", what, k, (long long)-lo); return TSR_EARG; }
    if (with_data == 2)                            /* an output: every element its own address (no write races) */
        for (int32_t d = 0; d < a->ndim; d++)
            if (a->shape[d] > 1 && a->strides[d] == 0) {
                fn_set_error("%s %d: zero stride on axis %d of an output (its elements would alias)", what, k, (int)d);
                return TSR_EARG;
            }
    if ((uint64_t)hi > UINTPTR_MAX - (uintptr_t)a->data) { fn_set_error("%s %d: the elements wrap the address space", what, k); return TSR_EARG; }
    /* kernels load elements through typed pointers: every element address must be aligned to its type */
    const int64_t al = el > 8 ? 8 : el;
    if (al > 1) {
        int64_t mis = (int64_t)(((uintptr_t)a->data + (uintptr_t)a->offset) % (uintptr_t)al);
        for (int32_t d = 0; d < a->ndim && mis == 0; d++)
            if (a->shape[d] > 1 && a->strides[d] % al != 0) mis = 1;
        if (mis) { fn_set_error("%s %d: elements are not aligned to %lld bytes", what, k, (long long)al); return TSR_EARG; }
    }
    return TSR_OK;
}

static int gate_arg(const tsr_arg *a, int k, int depth)
{
    switch (a->kind) {
    case 0: case 4:
        break;
    case 1:
        if (a->flags & ~1) { fn_set_error("argument %d: undefined flag bits 0x%x for a number", k, (unsigned)a->flags); return TSR_EARG; }
        if ((a->flags & 1) && a->num != (double)a->ival) {
            fn_set_error("argument %d: integer value %lld and its double %.17g disagree", k, (long long)a->ival, a->num);
            return TSR_EARG;
        }
        return TSR_OK;
    case 2:
        if (!a->str) { fn_set_error("argument %d: string is NULL", k); return TSR_EARG; }
        if (!memchr(a->str, '\0', (size_t)FN_GATE_MAXSTR)) { fn_set_error("argument %d: string longer than %lld bytes", k, (long long)FN_GATE_MAXSTR); return TSR_EARG; }
        break;
    case 3:
        if (a->flags & ~2) { fn_set_error("argument %d: undefined flag bits 0x%x for an array", k, (unsigned)a->flags); return TSR_EARG; }
        return fn_gate_array(&a->arr, 1, "argument", k);
    case 5:
        if (depth >= FN_GATE_MAXDEPTH) { fn_set_error("argument %d: sequences nested deeper than %d", k, FN_GATE_MAXDEPTH); return TSR_EARG; }
        if (a->count < 0 || a->count > FN_GATE_MAXSEQ) { fn_set_error("argument %d: sequence length %lld out of range", k, (long long)a->count); return TSR_EARG; }
        if (a->count > 0 && !a->items) { fn_set_error("argument %d: sequence items are NULL", k); return TSR_EARG; }
        for (int64_t j = 0; j < a->count; j++) {
            const int rc = gate_arg(&a->items[j], k, depth + 1);
            if (rc != TSR_OK) return rc;
        }
        break;
    default:
        fn_set_error("argument %d: unknown kind %d", k, (int)a->kind);
        return TSR_EARG;
    }
    if (a->flags != 0) { fn_set_error("argument %d: undefined flag bits 0x%x for kind %d", k, (unsigned)a->flags, (int)a->kind); return TSR_EARG; }
    return TSR_OK;
}

int fn_gate_args(int nargs, const tsr_arg *args)
{
    if (nargs < 0) { fn_set_error("negative argument count %d", nargs); return TSR_EARG; }
    if (nargs > 0 && !args) { fn_set_error("arguments are NULL"); return TSR_EARG; }
    for (int k = 0; k < nargs; k++) {
        const int rc = gate_arg(&args[k], k, 0);
        if (rc != TSR_OK) return rc;
    }
    return TSR_OK;
}

int fn_gate_strided(int32_t ndim, const int64_t *shape, int nop, void *const *data, const int64_t *strides, const int64_t *elsize,
                    int nout)
{
    int64_t size;
    int rc = gate_shape(ndim, shape, "operand set", 0, &size);
    if (rc != TSR_OK) return rc;
    if (nop < 0 || (nop > 0 && !data)) { fn_set_error("operands are NULL"); return TSR_EARG; }
    if (ndim > 0 && nop > 0 && !strides) { fn_set_error("strides are NULL"); return TSR_EARG; }
    for (int o = 0; o < nop; o++) {
        if (!data[o]) { fn_set_error("operand %d: data is NULL", o); return TSR_EARG; }
        if (size == 0) continue;
        /* kernels load operands through typed pointers: the first element and every step must be aligned */
        const int64_t al = elsize[o] > 8 ? 8 : elsize[o];
        if (al > 1) {
            int mis = ((uintptr_t)data[o] % (uintptr_t)al) != 0;
            for (int32_t d = 0; d < ndim && !mis; d++)
                if (shape[d] > 1 && strides[(int64_t)o * ndim + d] % al != 0) mis = 1;
            if (mis) { fn_set_error("operand %d: elements are not aligned to %lld bytes", o, (long long)al); return TSR_EARG; }
        }
        if (ndim == 0) continue;
        if (o >= nop - nout)                       /* an output: every element its own address (no write races) */
            for (int32_t d = 0; d < ndim; d++)
                if (shape[d] > 1 && strides[(int64_t)o * ndim + d] == 0) {
                    fn_set_error("operand %d: zero stride on axis %d of an output (its elements would alias)", o, (int)d);
                    return TSR_EARG;
                }
        int64_t lo, hi;
        if (!gate_span(ndim, shape, strides + (int64_t)o * ndim, elsize[o], &lo, &hi)
            || (uint64_t)hi > UINTPTR_MAX - (uintptr_t)data[o] || (uint64_t)0 - (uint64_t)lo > (uintptr_t)data[o]) {
            fn_set_error("operand %d: strides overflow the address range", o);
            return TSR_EARG;
        }
    }
    return TSR_OK;
}
