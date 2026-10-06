/*
 * The function registry (ADR 0011): special functions, statistics and the other SciPy-style routines.
 *
 * Every callable is described once, here in the kernel. Both bindings read the registry at run time
 * (tsr_fn_count / tsr_fn_name / tsr_fn_info) and expose each entry as a PHP static method, so a new
 * function reaches the FFI package and the extension together, with the same name, arguments and results.
 *
 * Kinds
 *   FN_UFUNC   element-wise over broadcast operands: nin float64 inputs -> nout float64 outputs, one call of
 *              `elem(ctx, in, out)` per element (float32 in -> float32 out, computed in double as SciPy does).
 *   FN_GUFUNC  generalised ufunc: each input contributes a "core" (the axes selected by its mask, gathered into
 *              a contiguous float64 buffer); `core(...)` maps the cores to the output cores. The remaining
 *              ("loop") axes of all inputs broadcast together, exactly like NumPy gufuncs.
 *   FN_DIST    a probability distribution (scipy.stats.rv_continuous / rv_discrete): methods pdf, cdf, ppf, ...
 *              are ufuncs over (x, shapes..., loc, scale); rvs draws from a PCG64 state.
 */
#ifndef TSR_FN_H
#define TSR_FN_H

#include <math.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "internal.h"

#define FN_MAXOP 16                         /* operands of one element-wise call (inputs + outputs) */
#define FN_MAXPARAM 16

enum { FN_UFUNC = 1, FN_GUFUNC = 2, FN_DIST = 3, FN_ROUTINE = 4, FN_RANDOM = 5 };

/* FN_RANDOM: a numpy.random.Generator method. One variate per output element, in C order, from parameters p
   (the broadcast array arguments, then the enum parameters); bitgen_t is numpy's bit-generator interface. */
struct bitgen;
typedef double (*fn_rdraw)(struct bitgen *bg, const double *p);
typedef int64_t (*fn_rdraw_i)(struct bitgen *bg, const double *p);
typedef float (*fn_rdraw_f)(struct bitgen *bg, const double *p);
/* checks one element's parameters before anything is drawn; returns an error message or NULL */
typedef const char *(*fn_rcheck)(const double *p);

/* per-element function: in[nin] -> out[nout] */
typedef void (*fn_elem)(const void *ctx, const double *in, double *out);

/* core of a gufunc: x[i] points to n[i] gathered values of input i (private copies: may be modified);
   out[j] has nout[j] slots. Return TSR_OK or an error (message via fn_set_error). */
typedef int (*fn_core)(const void *ctx, const double *const *x, const int64_t *n, const double *p,
                       double *const *out, const int64_t *nout, void *scratch, unsigned flags);
/* flags passed to cores */
#define FN_SEQUENTIAL 1u   /* NumPy would reduce this core sequentially (the core axis is not the innermost one) */
#define FN_INT64 2u        /* CORE_INT64 mode: input 0 and output 0 are int64_t (see CORE_INT64) */
/* a routine: any arguments, results allocated by the routine (fn_result_array) */
typedef int (*fn_routine)(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres);
/* the language boundary: fn_core.c (C) calls every gufunc core and routine through these. The C++ versions
   (cxx/fn_guard.cpp) catch any C++ exception, which must never unwind through C frames or into PHP (that
   terminates the process), and turn it into an error code and message. */
int fn_call_core(fn_core f, const void *ctx, const double *const *x, const int64_t *n, const double *p,
                 double *const *out, const int64_t *out_n, void *scratch, unsigned flags);
int fn_call_routine(fn_routine f, const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres);
/* scratch bytes the core needs for input lengths n (may be NULL: none) */
typedef int64_t (*fn_scratch)(const int64_t *n, const double *p);

/* output core shape rules: base (bits 0-7) | CORE_ADD(j) (+1 when params[j] is true) | CORE_MINUS1 | CORE_FIRST */
enum { CORE_SCALAR = 0, CORE_LIKE0 = 1, CORE_LIKE1 = 2, CORE_LIKE2 = 3, CORE_PARAM = 16 /* + param index */,
       CORE_FIXED = 64 /* + length */ };
#define CORE_ADD(j) (((j) + 1) << 8)
#define CORE_FIRST 0x1000
#define CORE_MINUS1 0x2000
#define CORE_INPLACE 0x4000   /* a 1-D output core replaces input 0's (single) core axis in place (cumsum, partition) */
#define CORE_FLAT 0x8000      /* the output core is input 0's core flattened to 1-D (axis=None ravels, as NumPy) */
#define CORE_INTLIKE 0x10000  /* the output is int64 when input 0 is an integer array (nansum, nanmin, ptp ...) */
#define CORE_NOBOOL 0x40000   /* a boolean input 0 is a TypeError (numpy.ptp: boolean subtract is not supported) */
#define CORE_INT64 0x20000    /* with CORE_INTLIKE: the core also has an exact int64 mode. On an integer input 0 the
                                 engine gathers it as int64 (the double* core pointers then point at int64_t),
                                 passes FN_INT64 in flags, and stores output 0's slots as int64: NumPy's integer
                                 arithmetic (wrap-around on overflow) instead of float64 rounding above 2^53 */

/* axis rules of a generalised ufunc (fn_def.axis_rules) */
enum { AX_SINGLE = 1 /* an int or None; a tuple of axes is a TypeError, as in NumPy */,
       AX_NOKEEP = 2 /* no keepdims argument */,
       AX_NONE_1D = 4 /* axis=None only for 1-D input (numpy.cumulative_sum) */,
       AX_WEIGHTS = 8 /* input 1 is numpy.average's weights: 0-d (none), a's shape, or a's shape along the axes */,
       AX_IN0_1D = 16 /* input 0 must be 1-D (numpy.searchsorted's sorted array) */ };

/* nan_policy support flags */
enum { NANP_NONE = 0, NANP_ALL = 1, NANP_PAIRED = 2 /* omit drops a position when any input has NaN there */ };

typedef struct {
    const char *name;                        /* "special.gammainc", "stats.median", ... */
    int kind;
    int nin, nout;
    const char *args;                        /* input names, comma separated: "a, x" */
    const char *outs;                        /* output names: "statistic, pvalue" */
    const char *params;                      /* gufunc parameters: "ddof=0, method=linear|lower|higher" (enum: '|') */
    const char *doc;
    /* FN_UFUNC */
    fn_elem elem;
    fn_elem elem_int;                        /* the loop for integer arguments (NumPy picks it for integer arrays) */
    unsigned int_args;                       /* bit k: argument k is an integer in elem_int */
    /* FN_GUFUNC */
    fn_core core;
    fn_scratch scratch;
    int out_core[4];                         /* CORE_* per output */
    int out_int;                             /* bit j: output j is int64 */
    int out_bool;                            /* bit j: output j is bool */
    fn_routine routine;                      /* FN_ROUTINE */
    const char *axis_default;                /* "None" (flatten, numpy), "0" / "-1" (scipy.stats), or "hidden": no axis/keepdims
                                                arguments (whole-array cores, element-shaped result: digitize, isin) */
    int nanpolicy;                           /* NANP_ALL: accepts nan_policy=propagate|omit|raise */
    int ninputs_core;                        /* inputs that take the axis (the rest are whole-array cores) */
    const void *ctx;
    unsigned axis_rules;                     /* AX_* */
    /* FN_RANDOM */
    fn_rdraw rdraw;                          /* float64 variates */
    fn_rdraw_i rdraw_i;                      /* int64 variates (discrete distributions) */
    fn_rdraw_f rdraw_f;                      /* float32 variates (dtype=float32), may be NULL */
    const char *cons;                        /* one constraint code per array argument (RC_* below) */
    fn_rcheck rcheck;                        /* extra per-element check, may be NULL */
    /* FN_GUFUNC: output 0 keeps input 0's integer or bool dtype when the enum params[keep->param] is one of
       keep->mask's bits (numpy.quantile's discrete methods); with any other value a bool input 0 is a TypeError */
    const struct fn_keep *keep;
    /* FN_UFUNC complex loop: in/out are interleaved complex128 (2 doubles per operand). The engine runs this on
       complex128 operands when a backend forwards them; NULL means the ufunc has no complex loop (DTypeError). */
    fn_elem elem_cplx;
} fn_def;
typedef struct fn_keep { int param; unsigned mask; } fn_keep;

/* numpy's parameter constraints (numpy/random/_common.pyx), one character per argument */
#define RC_NONE '.'
#define RC_NON_NEGATIVE 'n'       /* "x < 0" (NaN passes) */
#define RC_POSITIVE 'p'           /* "x <= 0" (NaN passes) */
#define RC_POSITIVE_NOT_NAN 'P'
#define RC_BOUNDED_0_1 'b'        /* 0 <= x <= 1 */
#define RC_BOUNDED_GT_0_1 'g'     /* 0 < x <= 1 */
#define RC_BOUNDED_LT_0_1 'l'     /* 0 <= x < 1 */
#define RC_GT_1 '1'
#define RC_POISSON 'L'            /* 0 <= x <= POISSON_LAM_MAX */

/* the FN_RANDOM fields, then fn_def.keep */
#define FN_NO_RANDOM5 NULL, NULL, NULL, NULL, NULL
#define FN_NO_RANDOM FN_NO_RANDOM5, NULL
#define RANDOM(NAME, NIN, ARGS, PARAMS, DRAW, DRAWI, DRAWF, CONS, CHECK, DOC) \
    {NAME, FN_RANDOM, NIN, 1, ARGS, "out", PARAMS, DOC, NULL, NULL, 0, NULL, NULL, {0}, 0, 0, NULL, NULL, 0, 0, NULL, 0, \
     DRAW, DRAWI, DRAWF, CONS, CHECK, NULL}

/* ---------------------------------------------------------------- distributions */

/* methods of a distribution: pdf..isf take x (or q) first; stats takes the moments mask (1 m, 2 v, 4 s, 8 k)
   first and returns mean, var, skew, kurtosis; moment takes the order first; entropy and support take only
   the parameters. The parameters are the shape parameters, loc and (continuous only) scale. */
enum { DM_PDF, DM_LOGPDF, DM_CDF, DM_LOGCDF, DM_SF, DM_LOGSF, DM_PPF, DM_ISF, DM_STATS, DM_ENTROPY, DM_SUPPORT,
       DM_MOMENT, DM_COUNT };

typedef struct fn_dist fn_dist;
/* a registered distribution: the description the registry serves; the implementation is the C++ object in
   `extra` (a tsd::Cont or tsd::Disc, cxx/stats_dist.hpp) */
struct fn_dist {
    const char *name;                        /* scipy name: "norm", "weibull_min" */
    const char *shapes;                      /* "a, b" or "" */
    int nshape;
    int discrete;
    const char *doc;
    double a, b;                             /* support of the standard form */
    const void *extra;
};

/* ---------------------------------------------------------------- module tables */

typedef struct { const fn_def *defs; int n; } fn_table;
typedef struct { const fn_dist *dists; int n; } fn_dist_table;

extern const fn_table *const TSR_FN_TABLES[];
extern const int TSR_FN_NTABLES;
extern const fn_dist_table *const TSR_DIST_TABLES[];
extern const int TSR_DIST_NTABLES;

/* thread-local error message for the last failing call */
void fn_set_error(const char *fmt, ...);

/* the distribution machinery (cxx/stats_dist.cpp) */
void dist_method(const fn_dist *d, int method, const double *in, double *out);
const char *dist_take_error(void);
int dist_rvs_array(const fn_dist *d, void *bitgen, int64_t n, const double *const *params, double *out);

/* helpers shared by modules */
#if defined(__GNUC__)
#define FN_NAN (__builtin_nan(""))
#define FN_INF (__builtin_inf())
#else
#define FN_NAN ((double)NAN)          /* <math.h>; MSVC has no __builtin_nan */
#define FN_INF ((double)INFINITY)
#endif

#define UFUNC(NAME, NIN, NOUT, ARGS, OUTS, FN, DOC) \
    {NAME, FN_UFUNC, NIN, NOUT, ARGS, OUTS, "", DOC, FN, NULL, 0, NULL, NULL, {0}, 0, 0, NULL, NULL, 0, 0, NULL, 0, FN_NO_RANDOM}
#define UFUNCI(NAME, NIN, NOUT, ARGS, OUTS, FN, FNI, MASK, DOC) \
    {NAME, FN_UFUNC, NIN, NOUT, ARGS, OUTS, "", DOC, FN, FNI, MASK, NULL, NULL, {0}, 0, 0, NULL, NULL, 0, 0, NULL, 0, FN_NO_RANDOM}
/* a ufunc with a complex128 loop FNC (in/out interleaved complex): scipy.special jv/yv/iv/kv accept complex z */
#define UFUNC_C(NAME, NIN, NOUT, ARGS, OUTS, FN, FNI, MASK, FNC, DOC) \
    {NAME, FN_UFUNC, NIN, NOUT, ARGS, OUTS, "", DOC, FN, FNI, MASK, NULL, NULL, {0}, 0, 0, NULL, NULL, 0, 0, NULL, 0, FN_NO_RANDOM, FNC}
/* generalised ufunc: C0..C3 output core rules, OUTINT / OUTBOOL output dtype bits, AXIS default ("None", "0", "-1"),
   NANP nan_policy support, NCORE inputs that take the axis (the others are whole-array cores) */
#define GUFUNC(NAME, NIN, NOUT, ARGS, OUTS, PARAMS, CORE, SCRATCH, C0, C1, C2, C3, OUTINT, OUTBOOL, AXIS, NANP, NCORE, CTX, DOC) \
    {NAME, FN_GUFUNC, NIN, NOUT, ARGS, OUTS, PARAMS, DOC, NULL, NULL, 0, CORE, SCRATCH, {C0, C1, C2, C3}, OUTINT, OUTBOOL, NULL, AXIS, NANP, NCORE, CTX, 0, FN_NO_RANDOM}
/* GUFUNC whose output 0 keeps an integer or bool input's dtype for some parameter values (fn_keep) */
#define GUFUNC_K(NAME, NIN, NOUT, ARGS, OUTS, PARAMS, CORE, SCRATCH, C0, C1, C2, C3, OUTINT, OUTBOOL, AXIS, NANP, NCORE, CTX, KEEP, DOC) \
    {NAME, FN_GUFUNC, NIN, NOUT, ARGS, OUTS, PARAMS, DOC, NULL, NULL, 0, CORE, SCRATCH, {C0, C1, C2, C3}, OUTINT, OUTBOOL, NULL, AXIS, NANP, NCORE, CTX, 0, FN_NO_RANDOM5, KEEP}
/* GUFUNC with axis rules AXR (AX_*) */
#define GUFUNC_R(NAME, NIN, NOUT, ARGS, OUTS, PARAMS, CORE, SCRATCH, C0, C1, C2, C3, OUTINT, OUTBOOL, AXIS, NANP, NCORE, CTX, AXR, DOC) \
    {NAME, FN_GUFUNC, NIN, NOUT, ARGS, OUTS, PARAMS, DOC, NULL, NULL, 0, CORE, SCRATCH, {C0, C1, C2, C3}, OUTINT, OUTBOOL, NULL, AXIS, NANP, NCORE, CTX, AXR, FN_NO_RANDOM}
/* the same with axis rules AXR (AX_*) */
#define GUFUNC_AX(NAME, NIN, NOUT, ARGS, OUTS, PARAMS, CORE, SCRATCH, C0, OUTINT, OUTBOOL, AXIS, NCORE, CTX, AXR, DOC) \
    {NAME, FN_GUFUNC, NIN, NOUT, ARGS, OUTS, PARAMS, DOC, NULL, NULL, 0, CORE, SCRATCH, {C0, 0, 0, 0}, OUTINT, OUTBOOL, NULL, AXIS, 0, NCORE, CTX, AXR, FN_NO_RANDOM}
/* routine: ARGS lists every argument with its default ("a, bins=10, range=None"); NOUT = most results.
   "arrays[]" marks a sequence argument (kind 5: a list of array-likes, numpy's `arrays`), "*xi" a variadic one
   (numpy's `*xi`: the positional arguments, as one kind-5 sequence; the arguments after it are keyword-only).
   OUTS "*name" returns one sequence (kind 6: numpy's list or tuple of arrays). "&a" marks an array the routine
   writes into (numpy.put, place, copyto, ...): the bindings pass it only when it is an NDArray that is not
   read-only, so the kernel may write through its strides. */
#define ROUTINE(NAME, NOUT, ARGS, OUTS, FN, CTX, DOC) \
    {NAME, FN_ROUTINE, 0, NOUT, ARGS, OUTS, "", DOC, NULL, NULL, 0, NULL, NULL, {0}, 0, 0, FN, NULL, 0, 0, CTX, 0, FN_NO_RANDOM}

/* allocate an array result (float64 / int64 / bool ...) of `shape`; NULL + TSR_ENOMEM on failure */
void *fn_result_array(tsr_result *r, int dtype, int32_t ndim, const int64_t *shape);
void fn_result_num(tsr_result *r, double v);
void fn_result_int(tsr_result *r, int64_t v);
int fn_result_str(tsr_result *r, const char *s);      /* a string result (kind 2): dtype names, ... */
void fn_results_free(tsr_result *res, int n);
/* a sequence result (kind 6) of n items, zeroed (kind 0); fill each with fn_result_array / _num / _int.
   Declare it with an OUTS string starting with '*' ("*arrays"). NULL + message on failure. */
tsr_result *fn_result_seq(tsr_result *r, int64_t n);
/* a routine argument as a contiguous float64 copy (tsr_alloc); *n elements; NULL on failure or when not an array */
double *fn_arg_doubles(const tsr_arg *a, int64_t *n);
void fn_free_doubles(double *p, int64_t n);
double tsr_psum(const double *x, int64_t n);           /* NumPy's pairwise summation (reduce.c) */
/* sum as numpy.add.reduce would: pairwise over a contiguous core, first + x1 + x2 ... otherwise */
static inline double fn_sum(const double *x, int64_t n, unsigned flags)
{
    if (n <= 0) return 0.0;
    if (!(flags & FN_SEQUENTIAL)) return tsr_psum(x, n);
    double acc = x[0];
    for (int64_t i = 1; i < n; i++) acc += x[i];
    return acc;
}

#ifdef __cplusplus
}
#endif
#endif
