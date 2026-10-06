/*
 * Tessero\Ext\Math — universal functions (ufuncs).
 *
 * A ufunc is an element-wise function over arrays of any shape, with NumPy's
 * broadcasting and type rules and an optional output array:
 *
 *   Math::sin($x);                    // new array
 *   Math::hypot($x, $y, out: $buf);   // written into $buf (no allocation)
 *   Math::sin($a, $a);                // in place
 *
 * Two families share one interface:
 *
 *   kernel ufuncs   the element-wise operations libtessero already has (sin,
 *                   exp, add, maximum, comparisons, ...). They run through the
 *                   same kernels as the NDArray methods and operators, so
 *                   Math::sin($a) and $a->sin() are identical to the bit.
 *
 *   loop ufuncs     cbrt, erf, gamma, logaddexp, fmax, copysign, fmod, ...
 *                   Tables of typed inner loops in libtessero (csrc/src/ufunc.c),
 *                   shared with the FFI package's Tessero\Math. This file
 *                   resolves the loop dtype (tsr_ufunc_resolve), casts the
 *                   inputs when needed, broadcasts with zero strides, picks the
 *                   output, and calls tsr_ufunc(), which coalesces dimensions
 *                   and splits large arrays across OpenMP threads.
 *
 * Results do not depend on the thread count.
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "Zend/zend_exceptions.h"
#include "libtessero/src/internal.h"     /* includes tessero.h */
#define TSR_TESSERO_H_INCLUDED
#include "php_tessero.h"
#include "libtessero/src/ops.h"

#include <math.h>
#include <string.h>

zend_class_entry *tsr_ce_math;

enum { D_F64 = 0, D_F32 = 1, D_I64 = 2, D_I32 = 3, D_U8 = 4, D_BOOL = 5, D_C128 = 6 };

enum { UF_NATIVE1, UF_NATIVE2, UF_LOOP };

typedef struct {
    const char *name;
    int kind;
    int op;                 /* kernel op for UF_NATIVE*, libtessero ufunc id for UF_LOOP */
    const char *summary;
} math_ufunc;

static const math_ufunc NATIVE[] = {
    /* ---- kernel ufuncs: one input */
    {"negative", UF_NATIVE1, U_NEG, "-x"},
    {"absolute", UF_NATIVE1, U_ABS, "|x| (modulus for complex)"},
    {"abs", UF_NATIVE1, U_ABS, "alias of absolute"},
    {"square", UF_NATIVE1, U_SQUARE, "x * x"},
    {"sign", UF_NATIVE1, U_SIGN, "-1, 0 or 1"},
    {"sqrt", UF_NATIVE1, U_SQRT, "square root"},
    {"exp", UF_NATIVE1, U_EXP, "e^x"},
    {"expm1", UF_NATIVE1, U_EXPM1, "e^x - 1, accurate near 0"},
    {"log", UF_NATIVE1, U_LOG, "natural logarithm"},
    {"log10", UF_NATIVE1, U_LOG10, "base-10 logarithm"},
    {"log2", UF_NATIVE1, U_LOG2, "base-2 logarithm"},
    {"log1p", UF_NATIVE1, U_LOG1P, "log(1 + x), accurate near 0"},
    {"sin", UF_NATIVE1, U_SIN, "sine (radians)"},
    {"cos", UF_NATIVE1, U_COS, "cosine (radians)"},
    {"tan", UF_NATIVE1, U_TAN, "tangent (radians)"},
    {"arcsin", UF_NATIVE1, U_ARCSIN, "inverse sine"},
    {"arccos", UF_NATIVE1, U_ARCCOS, "inverse cosine"},
    {"arctan", UF_NATIVE1, U_ARCTAN, "inverse tangent"},
    {"sinh", UF_NATIVE1, U_SINH, "hyperbolic sine"},
    {"cosh", UF_NATIVE1, U_COSH, "hyperbolic cosine"},
    {"tanh", UF_NATIVE1, U_TANH, "hyperbolic tangent"},
    {"floor", UF_NATIVE1, U_FLOOR, "round down"},
    {"ceil", UF_NATIVE1, U_CEIL, "round up"},
    {"rint", UF_NATIVE1, U_RINT, "round to nearest, ties to even"},
    {"reciprocal", UF_NATIVE1, U_RECIPROCAL, "1 / x"},
    {"isnan", UF_NATIVE1, U_ISNAN, "x is NaN (bool)"},
    {"isinf", UF_NATIVE1, U_ISINF, "x is +/-INF (bool)"},
    {"isfinite", UF_NATIVE1, U_ISFINITE, "x is finite (bool)"},
    {"invert", UF_NATIVE1, U_NOT, "bitwise NOT (logical NOT for bool)"},
    {"real", UF_NATIVE1, U_REAL, "real part"},
    {"imag", UF_NATIVE1, U_IMAG, "imaginary part"},
    {"conj", UF_NATIVE1, U_CONJ, "complex conjugate"},
    {"angle", UF_NATIVE1, U_ANGLE, "argument of a complex number"},
    /* ---- kernel ufuncs: two inputs */
    {"add", UF_NATIVE2, OP_ADD, "x + y"},
    {"subtract", UF_NATIVE2, OP_SUB, "x - y"},
    {"multiply", UF_NATIVE2, OP_MUL, "x * y"},
    {"divide", UF_NATIVE2, OP_DIV, "x / y (true division)"},
    {"power", UF_NATIVE2, OP_POW, "x ** y"},
    {"mod", UF_NATIVE2, OP_MOD, "remainder with the sign of y (Python %)"},
    {"remainder", UF_NATIVE2, OP_MOD, "alias of mod"},
    {"floorDivide", UF_NATIVE2, OP_FLOORDIV, "floor(x / y)"},
    {"maximum", UF_NATIVE2, OP_MAX, "element-wise maximum, NaN propagates"},
    {"minimum", UF_NATIVE2, OP_MIN, "element-wise minimum, NaN propagates"},
    {"arctan2", UF_NATIVE2, OP_ATAN2, "angle of (x=y2, y=x1): atan2(x1, x2)"},
    {"hypot", UF_NATIVE2, OP_HYPOT, "sqrt(x^2 + y^2) without overflow"},
    {"equal", UF_NATIVE2, OP_EQ, "x == y (bool)"},
    {"notEqual", UF_NATIVE2, OP_NE, "x != y (bool)"},
    {"less", UF_NATIVE2, OP_LT, "x < y (bool)"},
    {"lessEqual", UF_NATIVE2, OP_LE, "x <= y (bool)"},
    {"greater", UF_NATIVE2, OP_GT, "x > y (bool)"},
    {"greaterEqual", UF_NATIVE2, OP_GE, "x >= y (bool)"},
    {"logicalAnd", UF_NATIVE2, OP_AND, "x AND y (bool)"},
    {"logicalOr", UF_NATIVE2, OP_OR, "x OR y (bool)"},
    {"logicalXor", UF_NATIVE2, OP_XOR, "x XOR y (bool)"},
};
#define N_NATIVE (sizeof(NATIVE) / sizeof(NATIVE[0]))
#define MAX_UFUNCS 160

/* native entries then libtessero's loop ufuncs; filled once at MINIT, read-only afterwards */
static math_ufunc UFUNCS[MAX_UFUNCS];
static size_t N_UFUNCS;

static const math_ufunc *find_ufunc(zend_string *name)
{
    for (size_t i = 0; i < N_UFUNCS; i++)
        if (zend_string_equals_cstr(name, UFUNCS[i].name, strlen(UFUNCS[i].name))) return &UFUNCS[i];
    return NULL;
}

static int ufunc_nin(const math_ufunc *u)
{
    return u->kind == UF_NATIVE2 || (u->kind == UF_LOOP && tsr_ufunc_nin(u->op) == 2) ? 2 : 1;
}

/* ================================================================ engine */

static inline char *first(const tsr_array *a) { return (char *)a->data + a->offset; }

/* Does writing `out` (with strides ost) clobber input `in` (strides ist) before it is read? */
static int overlaps(const tsr_array *out, const int64_t *ost, const tsr_array *in, const int64_t *ist, int32_t nd)
{
    if (in->data != out->data) return 0;                       /* different blocks (or different mappings) */
    if (in->offset != out->offset) return 1;
    for (int d = 0; d < nd; d++) if (ist[d] != ost[d] && out->shape[d] > 1) return 1;
    return 0;                                                  /* identical layout: in place is fine */
}

static int apply_loop_ufunc(const math_ufunc *u, zval *x, zval *y, tsr_obj *dst, zval *rv)
{
    const int nin = ufunc_nin(u);
    int dt = tsr_result_dtype(x, nin == 2 ? y : NULL);
    if (dt < 0) {
        if (!EG(exception)) zend_throw_exception(tsr_ce_dtype_exception, "Math: unsupported operand", 0);
        return FAILURE;
    }
    if (dt == D_C128) {
        zend_throw_exception_ex(tsr_ce_dtype_exception, 0, "Math::%s() does not support complex128", u->name);
        return FAILURE;
    }
    int ldt, odt, native_op = 0;
    const int how = tsr_ufunc_resolve(u->op, dt, &ldt, &odt, &native_op);
    if (how < 0) {
        zend_throw_exception_ex(tsr_ce_dtype_exception, 0, "Math::%s() has no loop for %s", u->name, tsr_dtype_name(dt));
        return FAILURE;
    }
    if (how == 2) return tsr_binary_into(native_op, x, y, dst, rv);    /* fmax on integers is maximum */
    if (how == 1) {                                                   /* trunc on integers is the identity */
        zval A;
        if (tsr_operand(x, dt, &A) == FAILURE) return FAILURE;
        int r;
        if (dst) {
            r = tsr_cast_into(&Z_TSR(A)->a, dst, "out");
            if (r == SUCCESS) ZVAL_OBJ_COPY(rv, &dst->std);
        } else {
            tsr_array *a = &Z_TSR(A)->a;
            r = tsr_ndarray_new(rv, a->dtype, a->ndim, a->shape, 0);
            if (r == SUCCESS && tsr_array_size(a) > 0)
                tsr_copy(a->dtype, a->dtype, a->ndim, a->shape, first(a), a->strides, first(&Z_TSR_P(rv)->a), Z_TSR_P(rv)->a.strides);
        }
        zval_ptr_dtor(&A);
        return r;
    }
    int ret = FAILURE;
    zval ops[2], cast[2], T;
    ZVAL_UNDEF(&ops[0]); ZVAL_UNDEF(&ops[1]); ZVAL_UNDEF(&cast[0]); ZVAL_UNDEF(&cast[1]); ZVAL_UNDEF(&T);
    zval *in[2] = {x, y};
    tsr_array a[2];
    for (int i = 0; i < nin; i++) {
        if (tsr_operand(in[i], ldt, &ops[i]) == FAILURE) goto out;
        if (Z_TSR(ops[i])->a.dtype != ldt) {
            if (tsr_ndarray_from_zval(&ops[i], ldt, &cast[i]) == FAILURE) goto out;
            a[i] = Z_TSR(cast[i])->a;
        } else {
            a[i] = Z_TSR(ops[i])->a;
        }
    }

    /* broadcast shape of the inputs */
    int32_t nd = a[0].ndim;
    int64_t shape[32];
    memcpy(shape, a[0].shape, sizeof(int64_t) * (size_t)nd);
    if (nin == 2 && tsr_broadcast_shape(a[0].ndim, a[0].shape, a[1].ndim, a[1].shape, &nd, shape) != 0) {
        char s1[160], s2[160];
        tsr_shape_str(s1, sizeof(s1), &a[0]);
        tsr_shape_str(s2, sizeof(s2), &a[1]);
        zend_throw_exception_ex(tsr_ce_shape_exception, 0, "Math::%s(): shapes %s and %s cannot be broadcast together", u->name, s1, s2);
        goto out;
    }
    int64_t ist[2][32];
    for (int i = 0; i < nin; i++) tsr_broadcast_strides(&a[i], nd, shape, ist[i]);

    /* output: straight into dst when shape and dtype match and no input is clobbered, else a new array */
    tsr_array *target = NULL;
    int direct = 0;
    if (dst) {
        if (tsr_require_writable(dst, "out") == FAILURE) goto out;
        direct = dst->a.dtype == odt && dst->a.ndim == nd && memcmp(dst->a.shape, shape, sizeof(int64_t) * (size_t)nd) == 0;
        for (int i = 0; direct && i < nin; i++) if (overlaps(&dst->a, dst->a.strides, &a[i], ist[i], nd)) direct = 0;
        if (direct) target = &dst->a;
    }
    if (!target) {
        if (tsr_ndarray_new(&T, odt, nd, shape, 0) == FAILURE) goto out;
        target = &Z_TSR(T)->a;
    }

    if (tsr_ufunc(u->op, ldt, nd, shape, first(&a[0]), ist[0], nin == 2 ? first(&a[1]) : NULL, nin == 2 ? ist[1] : NULL,
                  first(target), target->strides) != 0) {
        zend_throw_exception_ex(tsr_ce_exception, 0, "Math::%s(): kernel error", u->name);
        goto out;
    }

    if (dst && !direct) {
        if (tsr_cast_into(target, dst, "out") == FAILURE) goto out;
        ZVAL_OBJ_COPY(rv, &dst->std);
    } else if (dst) {
        ZVAL_OBJ_COPY(rv, &dst->std);
    } else {
        ZVAL_COPY(rv, &T);
    }
    ret = SUCCESS;
out:
    zval_ptr_dtor(&ops[0]); zval_ptr_dtor(&ops[1]); zval_ptr_dtor(&cast[0]); zval_ptr_dtor(&cast[1]); zval_ptr_dtor(&T);
    return ret;
}

static int apply_ufunc(const math_ufunc *u, zval *x, zval *y, tsr_obj *dst, zval *rv)
{
    switch (u->kind) {
    case UF_NATIVE1: return tsr_unary_into(u->op, x, dst, rv);
    case UF_NATIVE2: return tsr_binary_into(u->op, x, y, dst, rv);
    default: return apply_loop_ufunc(u, x, y, dst, rv);
    }
}

/* ================================================================ PHP interface */

static int out_arg(zval *out, tsr_obj **dst, uint32_t argnum)
{
    *dst = NULL;
    if (out == NULL || Z_TYPE_P(out) == IS_NULL) return SUCCESS;
    if (Z_TYPE_P(out) != IS_OBJECT || Z_OBJCE_P(out) != tsr_ce_ndarray) {
        zend_argument_type_error(argnum, "must be of type ?Tessero\\Ext\\NDArray, %s given", zend_zval_type_name(out));
        return FAILURE;
    }
    *dst = Z_TSR_P(out);
    return SUCCESS;
}

/* Math::<unary>($x, ?NDArray $out = null) */
static PHP_METHOD(Math, unary)
{
    zval *x, *out = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(x)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(out)
    ZEND_PARSE_PARAMETERS_END();
    const math_ufunc *u = find_ufunc(EX(func)->common.function_name);
    tsr_obj *dst;
    if (!u || out_arg(out, &dst, 2) == FAILURE) RETURN_THROWS();
    if (apply_ufunc(u, x, NULL, dst, return_value) == FAILURE) RETURN_THROWS();
}

/* Math::<binary>($x, $y, ?NDArray $out = null) */
static PHP_METHOD(Math, binary)
{
    zval *x, *y, *out = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ZVAL(x)
        Z_PARAM_ZVAL(y)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(out)
    ZEND_PARSE_PARAMETERS_END();
    const math_ufunc *u = find_ufunc(EX(func)->common.function_name);
    tsr_obj *dst;
    if (!u || out_arg(out, &dst, 3) == FAILURE) RETURN_THROWS();
    if (apply_ufunc(u, x, y, dst, return_value) == FAILURE) RETURN_THROWS();
}

/* Math::ufuncs(): ['sin' => ['nin' => 1, 'engine' => 'kernel', 'summary' => ...], ...] */
static PHP_METHOD(Math, ufuncs)
{
    ZEND_PARSE_PARAMETERS_NONE();
    array_init_size(return_value, N_UFUNCS);
    for (size_t i = 0; i < N_UFUNCS; i++) {
        const math_ufunc *u = &UFUNCS[i];
        zval e;
        array_init_size(&e, 3);
        add_assoc_long(&e, "nin", ufunc_nin(u));
        add_assoc_string(&e, "engine", (u->kind == UF_NATIVE1 || u->kind == UF_NATIVE2) ? "kernel" : "loop");
        add_assoc_string(&e, "summary", u->summary);
        add_assoc_zval(return_value, u->name, &e);
    }
}

/* Math::apply('cbrt', $x) / Math::apply('fmax', $x, $y): call a ufunc by name. */
static PHP_METHOD(Math, apply)
{
    zend_string *name;
    zval *x, *y = NULL, *out = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_STR(name)
        Z_PARAM_ZVAL(x)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(y)
        Z_PARAM_ZVAL_OR_NULL(out)
    ZEND_PARSE_PARAMETERS_END();
    const math_ufunc *u = find_ufunc(name);
    if (!u) { zend_argument_value_error(1, "is not a ufunc (see Math::ufuncs())"); RETURN_THROWS(); }
    const int nin = ufunc_nin(u);
    if (nin == 2 && (y == NULL || Z_TYPE_P(y) == IS_NULL)) {
        zend_argument_count_error("Math::%s() takes two operands", u->name);
        RETURN_THROWS();
    }
    if (nin == 1 && out == NULL && y != NULL && Z_TYPE_P(y) == IS_OBJECT) out = y;   /* apply('sin', $x, $out) */
    tsr_obj *dst;
    if (out_arg(out, &dst, 4) == FAILURE) RETURN_THROWS();
    if (apply_ufunc(u, x, nin == 2 ? y : NULL, dst, return_value) == FAILURE) RETURN_THROWS();
}

ZEND_BEGIN_ARG_INFO_EX(ai_math_unary, 0, 0, 1)
    ZEND_ARG_INFO(0, x)
    ZEND_ARG_OBJ_INFO(0, out, Tessero\\Ext\\NDArray, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_math_binary, 0, 0, 2)
    ZEND_ARG_INFO(0, x)
    ZEND_ARG_INFO(0, y)
    ZEND_ARG_OBJ_INFO(0, out, Tessero\\Ext\\NDArray, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_math_none, 0, 0, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_math_apply, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
    ZEND_ARG_INFO(0, x)
    ZEND_ARG_INFO(0, y)
    ZEND_ARG_OBJ_INFO(0, out, Tessero\\Ext\\NDArray, 1)
ZEND_END_ARG_INFO()

static zend_function_entry math_fe[MAX_UFUNCS + 3];

void tsr_register_math(void)
{
    /* one static method per ufunc, all sharing the two handlers above (the handler looks up its own name) */
    N_UFUNCS = 0;
    for (size_t i = 0; i < N_NATIVE; i++) UFUNCS[N_UFUNCS++] = NATIVE[i];
    for (int id = 0; id < tsr_ufunc_count() && N_UFUNCS < MAX_UFUNCS; id++)
        UFUNCS[N_UFUNCS++] = (math_ufunc){tsr_ufunc_name(id), UF_LOOP, id, tsr_ufunc_summary(id)};

    size_t k = 0;
    for (size_t i = 0; i < N_UFUNCS; i++) {
        if (ufunc_nin(&UFUNCS[i]) == 2) {
            const zend_function_entry e[] = { TSR_STATIC_FE(UFUNCS[i].name, zim_Math_binary, ai_math_binary) };
            math_fe[k++] = e[0];
        } else {
            const zend_function_entry e[] = { TSR_STATIC_FE(UFUNCS[i].name, zim_Math_unary, ai_math_unary) };
            math_fe[k++] = e[0];
        }
    }
    const zend_function_entry tail[] = {
        TSR_STATIC_FE("ufuncs", zim_Math_ufuncs, ai_math_none)
        TSR_STATIC_FE("apply", zim_Math_apply, ai_math_apply)
        PHP_FE_END
    };
    memcpy(&math_fe[k], tail, sizeof(tail));

    zend_class_entry ce;
    INIT_NS_CLASS_ENTRY(ce, "Tessero\\Ext", "Math", math_fe);
    tsr_ce_math = zend_register_internal_class(&ce);
    /* a namespace of functions: static methods only, never instantiated */
    tsr_ce_math->ce_flags |= ZEND_ACC_EXPLICIT_ABSTRACT_CLASS | ZEND_ACC_NO_DYNAMIC_PROPERTIES;
}
