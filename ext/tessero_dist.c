/*
 * Tessero\Ext\Distribution: a frozen scipy.stats distribution (Stats::norm(loc: 1, scale: 2)).
 *
 * The object stores the registry id and the parameters; every method is one element-wise call into the
 * kernel's distribution machinery (tessero_fn.c, ADR 0011), with the parameters broadcast against x.
 * The FFI package's Tessero\Stats\Distribution has the same methods and results.
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "Zend/zend_exceptions.h"
#include "Zend/zend_interfaces.h"
#include "libtessero/src/internal.h"
#define TSR_TESSERO_H_INCLUDED
#include "php_tessero.h"

#include <math.h>

enum { D_F64_ID = 0, D_I64_ID = 2 };

zend_class_entry *tsr_ce_distribution;
static zend_object_handlers dist_handlers;

typedef struct {
    int id;
    int nparams;
    zval params[16];
    char label[64];
    zend_object std;
} tsr_dist;

static inline tsr_dist *dist_from(zend_object *o) { return (tsr_dist *)((char *)o - XtOffsetOf(tsr_dist, std)); }
#define Z_DIST_P(zv) dist_from(Z_OBJ_P(zv))

enum { DM_PDF, DM_LOGPDF, DM_CDF, DM_LOGCDF, DM_SF, DM_LOGSF, DM_PPF, DM_ISF, DM_STATS, DM_ENTROPY, DM_SUPPORT, DM_MOMENT };

static zend_object *dist_create(zend_class_entry *ce)
{
    tsr_dist *d = zend_object_alloc(sizeof(tsr_dist), ce);
    d->id = -1;
    d->nparams = 0;
    zend_object_std_init(&d->std, ce);
    d->std.handlers = &dist_handlers;
    return &d->std;
}

static void dist_free(zend_object *o)
{
    tsr_dist *d = dist_from(o);
    for (int k = 0; k < d->nparams; k++) zval_ptr_dtor(&d->params[k]);
    zend_object_std_dtor(o);
}

void tsr_distribution_new(zval *rv, int id, const char *label, zval *args, int nargs)
{
    int nin = 0, nout = 0;
    /* entropy takes exactly the parameters: shapes, loc[, scale] */
    if (tsr_fn_arity(id, DM_ENTROPY, &nin, &nout) != 0) { zend_throw_error(NULL, "%s: not a distribution", label); return; }
    if (nargs > nin || nargs > 16) { zend_argument_count_error("%s() takes at most %d arguments", label, nin); return; }
    object_init_ex(rv, tsr_ce_distribution);
    tsr_dist *d = Z_DIST_P(rv);
    d->id = id;
    snprintf(d->label, sizeof d->label, "%s", label);
    d->nparams = nin;
    for (int k = 0; k < nin; k++) {
        if (k < nargs && Z_TYPE(args[k]) != IS_UNDEF) {
            zval *z = &args[k];
            ZVAL_DEREF(z);
            ZVAL_COPY(&d->params[k], z);
        } else {
            ZVAL_LONG(&d->params[k], 0);             /* loc = 0 (scale is set below) */
        }
    }
    /* scale defaults to 1 for continuous distributions (their last parameter) */
    char buf[512];
    int64_t len = tsr_fn_info(id, buf, sizeof buf);
    const int is_discrete = len > 0 && strstr(buf, "\"discrete\":true") != NULL;
    if (!is_discrete && nargs < nin) { zval_ptr_dtor(&d->params[nin - 1]); ZVAL_LONG(&d->params[nin - 1], 1); }
    /* required shape parameters */
    const int nshape = nin - (is_discrete ? 1 : 2);
    if (nargs < nshape) {
        zval_ptr_dtor(rv);
        ZVAL_UNDEF(rv);
        zend_argument_count_error("%s() requires %d shape parameter(s)", label, nshape);
    }
}

/* poisson_binom: a distribution whose shape parameter is the whole probability vector p, which the broadcast
 * distribution machinery cannot express. It reuses this Distribution object with the sentinel id PB_ID, storing p
 * in params[0] and loc in params[1], and dispatches to the kernel's tsr_poisson_binom entry points instead of
 * tsr_fn_ufunc. (The FFI package's Tessero\Stats\PoissonBinom is the same facade.) */
#define PB_ID (-7)
extern int tsr_poisson_binom(int method, const double *xv, int64_t nx, const double *p, int64_t n, double loc, double *out);
extern int tsr_poisson_binom_rvs(uint64_t *state, const double *p, int64_t n, double loc, int64_t nsamp, double *out);

/* the probability vector of a poisson_binom pseudo-distribution; `keep` holds the contiguous array alive. */
static int pb_prob(tsr_dist *d, double **pp, int64_t *pn, zval *keep)
{
    zval tmp;
    if (tsr_ndarray_from_zval(&d->params[0], D_F64_ID, &tmp) == FAILURE) return FAILURE;
    if (tsr_ndarray_contiguous_as(&tmp, D_F64_ID, keep) == FAILURE) { zval_ptr_dtor(&tmp); return FAILURE; }
    zval_ptr_dtor(&tmp);
    tsr_array *a = &Z_TSR(*keep)->a;
    int64_t nn = 1;
    for (int32_t i = 0; i < a->ndim; i++) nn *= a->shape[i];
    *pp = (double *)((char *)a->data + a->offset);
    *pn = nn;
    return SUCCESS;
}

/* every poisson_binom method but rvs: mirrors disc_method through tsr_poisson_binom. */
static void pb_call(tsr_dist *d, int method, zval *x, zval *rv)
{
    double *p;
    int64_t n;
    zval pk;
    if (pb_prob(d, &p, &n, &pk) == FAILURE) { zend_throw_error(NULL, "Stats::poissonBinom(): invalid probabilities"); return; }
    const double loc = zval_get_double(&d->params[1]);
    if (method == DM_STATS) {
        double out[4] = {0, 0, 0, 0};
        double mask = x ? zval_get_double(x) : 15.0;
        tsr_poisson_binom(DM_STATS, &mask, 1, p, n, loc, out);
        array_init(rv);
        add_assoc_double(rv, "mean", out[0]);
        add_assoc_double(rv, "var", out[1]);
        add_assoc_double(rv, "skew", out[2]);
        add_assoc_double(rv, "kurtosis", out[3]);
        zval_ptr_dtor(&pk);
        return;
    }
    if (method == DM_SUPPORT || method == DM_ENTROPY) {
        double out[2] = {0, 0};
        tsr_poisson_binom(method, NULL, 0, p, n, loc, out);
        if (method == DM_ENTROPY) { ZVAL_DOUBLE(rv, out[0]); }
        else { array_init(rv); add_assoc_double(rv, "a", out[0]); add_assoc_double(rv, "b", out[1]); }
        zval_ptr_dtor(&pk);
        return;
    }
    if (x) ZVAL_DEREF(x);
    if (x && (Z_TYPE_P(x) == IS_LONG || Z_TYPE_P(x) == IS_DOUBLE)) {
        double xv = zval_get_double(x), out = 0;
        tsr_poisson_binom(method, &xv, 1, p, n, loc, &out);
        ZVAL_DOUBLE(rv, out);
        zval_ptr_dtor(&pk);
        return;
    }
    zval xtmp, xc;
    if (!x || tsr_ndarray_from_zval(x, D_F64_ID, &xtmp) == FAILURE) { zval_ptr_dtor(&pk); zend_throw_error(NULL, "Stats::poissonBinom(): bad argument"); return; }
    if (tsr_ndarray_contiguous_as(&xtmp, D_F64_ID, &xc) == FAILURE) { zval_ptr_dtor(&xtmp); zval_ptr_dtor(&pk); zend_throw_error(NULL, "Stats::poissonBinom(): bad argument"); return; }
    zval_ptr_dtor(&xtmp);
    tsr_array *xa = &Z_TSR(xc)->a;
    int64_t nx = 1;
    for (int32_t i = 0; i < xa->ndim; i++) nx *= xa->shape[i];
    zval res;
    if (tsr_ndarray_new(&res, D_F64_ID, xa->ndim, xa->shape, 0) == FAILURE) { zval_ptr_dtor(&xc); zval_ptr_dtor(&pk); return; }
    tsr_array *ra = &Z_TSR(res)->a;
    tsr_poisson_binom(method, (const double *)((char *)xa->data + xa->offset), nx, p, n, loc, (double *)((char *)ra->data + ra->offset));
    zval_ptr_dtor(&xc);
    zval_ptr_dtor(&pk);
    ZVAL_COPY_VALUE(rv, &res);
}

/* poisson_binom rvs: int64 variates of shape `size` from the generator's stream. */
static void pb_rvs(tsr_dist *d, zval *size, zval *rng, zval *rv)
{
    double *p;
    int64_t n;
    zval pk;
    if (pb_prob(d, &p, &n, &pk) == FAILURE) { zend_throw_error(NULL, "Stats::poissonBinom()->rvs(): invalid probabilities"); return; }
    const double loc = zval_get_double(&d->params[1]);
    int32_t nd = 0;
    int64_t shape[32];
    if (size && Z_TYPE_P(size) == IS_LONG) { nd = 1; shape[0] = Z_LVAL_P(size); }
    else if (size && Z_TYPE_P(size) == IS_ARRAY) {
        zval *e;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(size), e) { if (nd < 32) shape[nd++] = zval_get_long(e); } ZEND_HASH_FOREACH_END();
    }
    int64_t nsamp = 1;
    for (int32_t i = 0; i < nd; i++) nsamp *= shape[i];
    zval res;
    if (tsr_ndarray_new(&res, D_F64_ID, nd, shape, 0) == FAILURE) { zval_ptr_dtor(&pk); return; }
    tsr_array *ra = &Z_TSR(res)->a;
    int rc = tsr_poisson_binom_rvs(tsr_generator_state(rng), p, n, loc, nsamp, (double *)((char *)ra->data + ra->offset));
    zval_ptr_dtor(&pk);
    if (rc < 0) { zval_ptr_dtor(&res); zend_value_error("Stats::poissonBinom()->rvs(): failed"); return; }
    if (nd == 0) { double v = *(double *)((char *)ra->data + ra->offset); zval_ptr_dtor(&res); ZVAL_LONG(rv, (zend_long)v); return; }
    zval ints;
    if (tsr_ndarray_contiguous_as(&res, D_I64_ID, &ints) == FAILURE) { zval_ptr_dtor(&res); return; }
    zval_ptr_dtor(&res);
    ZVAL_COPY_VALUE(rv, &ints);
}

/* Stats::poissonBinom(p, loc = 0): a frozen poisson_binom distribution object. */
static void tsr_stats_poisson_binom(INTERNAL_FUNCTION_PARAMETERS)
{
    zval *p;
    double loc = 0.0;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(p)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(loc)
    ZEND_PARSE_PARAMETERS_END();
    object_init_ex(return_value, tsr_ce_distribution);
    tsr_dist *d = Z_DIST_P(return_value);
    d->id = PB_ID;
    snprintf(d->label, sizeof d->label, "Stats::poissonBinom");
    d->nparams = 2;
    ZVAL_DEREF(p);
    ZVAL_COPY(&d->params[0], p);
    ZVAL_DOUBLE(&d->params[1], loc);
}

/* Stats::poissonBinomPmf(k, p, loc = 0): the pmf without constructing a distribution (functional shortcut). */
static void tsr_stats_poisson_binom_pmf(INTERNAL_FUNCTION_PARAMETERS)
{
    zval *k, *p;
    double loc = 0.0;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ZVAL(k)
        Z_PARAM_ZVAL(p)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(loc)
    ZEND_PARSE_PARAMETERS_END();
    tsr_dist d;
    memset(&d, 0, sizeof d);
    d.id = PB_ID;
    d.nparams = 2;
    ZVAL_DEREF(p);
    ZVAL_COPY_VALUE(&d.params[0], p);
    ZVAL_DOUBLE(&d.params[1], loc);
    pb_call(&d, DM_PDF, k, return_value);
}

ZEND_BEGIN_ARG_INFO_EX(pb_ai, 0, 0, 1)
    ZEND_ARG_INFO(0, p)
    ZEND_ARG_INFO(0, loc)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(pbpmf_ai, 0, 0, 2)
    ZEND_ARG_INFO(0, k)
    ZEND_ARG_INFO(0, p)
    ZEND_ARG_INFO(0, loc)
ZEND_END_ARG_INFO()

/* called from tessero_fn.c to add the hand-written poisson_binom facades to Tessero\Ext\Stats */
void tsr_stats_register_pb(zend_function_entry *fes, int *nfe)
{
    zend_function_entry *fe = &fes[(*nfe)++];
    fe->fname = "poissonBinom";
    fe->handler = tsr_stats_poisson_binom;
    fe->arg_info = pb_ai;
    fe->num_args = 2;
    fe->flags = ZEND_ACC_PUBLIC | ZEND_ACC_STATIC;
    fe = &fes[(*nfe)++];
    fe->fname = "poissonBinomPmf";
    fe->handler = tsr_stats_poisson_binom_pmf;
    fe->arg_info = pbpmf_ai;
    fe->num_args = 3;
    fe->flags = ZEND_ACC_PUBLIC | ZEND_ACC_STATIC;
}

static void call_method(zval *self, int method, zval *x, zval *rv)
{
    tsr_dist *d = Z_DIST_P(self);
    if (d->id == PB_ID) { pb_call(d, method, x, rv); return; }
    zval args[17];
    int n = 0;
    if (x) ZVAL_COPY_VALUE(&args[n++], x);
    for (int k = 0; k < d->nparams; k++) ZVAL_COPY_VALUE(&args[n++], &d->params[k]);
    tsr_fn_dist_call(d->id, method, args, n, rv);
}

#define DIST_X_METHOD(NAME, M)                                                  \
    static PHP_METHOD(Distribution, NAME)                                       \
    {                                                                           \
        zval *x;                                                                \
        ZEND_PARSE_PARAMETERS_START(1, 1)                                       \
            Z_PARAM_ZVAL(x)                                                     \
        ZEND_PARSE_PARAMETERS_END();                                            \
        call_method(ZEND_THIS, M, x, return_value);                             \
    }

DIST_X_METHOD(pdf, DM_PDF)
DIST_X_METHOD(logpdf, DM_LOGPDF)
DIST_X_METHOD(pmf, DM_PDF)
DIST_X_METHOD(logpmf, DM_LOGPDF)
DIST_X_METHOD(cdf, DM_CDF)
DIST_X_METHOD(logcdf, DM_LOGCDF)
DIST_X_METHOD(sf, DM_SF)
DIST_X_METHOD(logsf, DM_LOGSF)
DIST_X_METHOD(ppf, DM_PPF)
DIST_X_METHOD(isf, DM_ISF)

/* stats(moments = 'mv'): one value, or the requested ones keyed mean, var, skew, kurtosis (in that order) */
static void stats_mask(zval *self, int mask, zval *rv)
{
    zval all, m;
    ZVAL_LONG(&m, mask);
    call_method(self, DM_STATS, &m, &all);
    if (EG(exception)) return;
    static const char *keys[4] = {"mean", "var", "skew", "kurtosis"};
    int count = 0;
    for (int k = 0; k < 4; k++) if (mask & (1 << k)) count++;
    if (count == 1) {
        for (int k = 0; k < 4; k++)
            if (mask & (1 << k)) { zval *v = zend_hash_str_find(Z_ARRVAL(all), keys[k], strlen(keys[k])); if (v) ZVAL_COPY(rv, v); }
        zval_ptr_dtor(&all);
        return;
    }
    array_init(rv);
    for (int k = 0; k < 4; k++) {
        if (!(mask & (1 << k))) continue;
        zval *v = zend_hash_str_find(Z_ARRVAL(all), keys[k], strlen(keys[k]));
        if (v) { Z_TRY_ADDREF_P(v); add_assoc_zval(rv, keys[k], v); }
    }
    zval_ptr_dtor(&all);
}

static PHP_METHOD(Distribution, stats)
{
    zend_string *moments = NULL;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR(moments)
    ZEND_PARSE_PARAMETERS_END();
    const char *m = moments ? ZSTR_VAL(moments) : "mv";
    int mask = 0;
    for (const char *p = m; *p; p++) {
        switch (*p) {
        case 'm': mask |= 1; break;
        case 'v': mask |= 2; break;
        case 's': mask |= 4; break;
        case 'k': mask |= 8; break;
        default: zend_argument_value_error(1, "must contain only 'm', 'v', 's', 'k'"); RETURN_THROWS();
        }
    }
    stats_mask(ZEND_THIS, mask, return_value);
}

static PHP_METHOD(Distribution, mean) { ZEND_PARSE_PARAMETERS_NONE(); stats_mask(ZEND_THIS, 1, return_value); }
static PHP_METHOD(Distribution, var) { ZEND_PARSE_PARAMETERS_NONE(); stats_mask(ZEND_THIS, 2, return_value); }

static PHP_METHOD(Distribution, std)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zval v;
    ZVAL_UNDEF(&v);
    stats_mask(ZEND_THIS, 2, &v);
    if (EG(exception)) RETURN_THROWS();
    if (Z_TYPE(v) == IS_DOUBLE) { RETURN_DOUBLE(sqrt(Z_DVAL(v))); }
    /* array result: element-wise sqrt through the NDArray method */
    zend_call_method(Z_OBJ(v), Z_OBJCE(v), NULL, "sqrt", sizeof("sqrt") - 1, return_value, 0, NULL, NULL);
    zval_ptr_dtor(&v);
}

static PHP_METHOD(Distribution, moment)
{
    zend_long order;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(order)
    ZEND_PARSE_PARAMETERS_END();
    zval o;
    ZVAL_DOUBLE(&o, (double)order);
    call_method(ZEND_THIS, DM_MOMENT, &o, return_value);
}

/* rvs(size = null, randomState = null): drawn from the generator's stream as scipy.stats draws from a
   numpy Generator; parameters broadcast as SciPy does (leading length-1 dimensions ignored) */
static PHP_METHOD(Distribution, rvs)
{
    zval *size = NULL, *rng = NULL;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(size)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(rng, tsr_ce_generator)
    ZEND_PARSE_PARAMETERS_END();
    tsr_dist *d = Z_DIST_P(ZEND_THIS);
    zval gen;
    ZVAL_UNDEF(&gen);
    if (!rng) {
        zend_call_method(NULL, tsr_ce_generator, NULL, "defaultrng", sizeof("defaultrng") - 1, &gen, 0, NULL, NULL);
        if (EG(exception)) RETURN_THROWS();
        rng = &gen;
    }
    if (d->id == PB_ID) {
        pb_rvs(d, size, rng, return_value);
        if (Z_TYPE(gen) != IS_UNDEF) zval_ptr_dtor(&gen);
        return;
    }
    zval ops[16];
    int nops = 0;
    int32_t bnd = 0;
    int64_t bshape[32];
    for (int k = 0; k < d->nparams; k++) {
        zval tmp;
        if (tsr_ndarray_from_zval(&d->params[k], D_F64_ID, &tmp) == FAILURE) goto fail;
        if (tsr_ndarray_contiguous_as(&tmp, D_F64_ID, &ops[nops]) == FAILURE) { zval_ptr_dtor(&tmp); goto fail; }
        zval_ptr_dtor(&tmp);
        tsr_array *a = &Z_TSR(ops[nops])->a;
        nops++;
        /* squeeze leading length-1 dimensions (a contiguous array keeps its layout) */
        while (a->ndim > 0 && a->shape[0] == 1) {
            memmove(a->shape, a->shape + 1, sizeof(int64_t) * (size_t)(a->ndim - 1));
            memmove(a->strides, a->strides + 1, sizeof(int64_t) * (size_t)(a->ndim - 1));
            a->ndim--;
        }
        if (nops == 1) { bnd = a->ndim; memcpy(bshape, a->shape, sizeof(int64_t) * (size_t)a->ndim); continue; }
        int32_t rnd;
        int64_t rsh[32];
        if (tsr_broadcast_shape(bnd, bshape, a->ndim, a->shape, &rnd, rsh) != 0) {
            zend_throw_exception(tsr_ce_shape_exception, "rvs: the parameters cannot be broadcast together", 0);
            goto fail;
        }
        bnd = rnd;
        memcpy(bshape, rsh, sizeof(int64_t) * (size_t)bnd);
    }
    int32_t nd = bnd;
    int64_t shape[32];
    memcpy(shape, bshape, sizeof(int64_t) * (size_t)bnd);
    if (size && Z_TYPE_P(size) != IS_NULL) {
        int32_t ns = 0;
        int64_t sz[32];
        if (Z_TYPE_P(size) == IS_LONG) sz[ns++] = Z_LVAL_P(size);
        else if (Z_TYPE_P(size) == IS_ARRAY && zend_hash_num_elements(Z_ARRVAL_P(size)) <= 32) {
            zval *e;
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(size), e) {
                if (Z_TYPE_P(e) != IS_LONG || Z_LVAL_P(e) < 0) { zend_argument_value_error(1, "must hold non-negative integers"); goto fail; }
                sz[ns++] = Z_LVAL_P(e);
            } ZEND_HASH_FOREACH_END();
        } else { zend_argument_type_error(1, "must be an int, a list of ints or null"); goto fail; }
        /* SciPy's _argcheck_rvs: pad the shorter with leading ones; each parameter dimension is 1 or matches */
        const int32_t n = bnd > ns ? bnd : ns;
        int64_t bb[32], ss[32];
        for (int32_t k = 0; k < n; k++) {
            bb[k] = k < n - bnd ? 1 : bshape[k - (n - bnd)];
            ss[k] = k < n - ns ? 1 : sz[k - (n - ns)];
            if (bb[k] != 1 && bb[k] != ss[k]) {
                zend_value_error("rvs: size does not match the broadcast shape of the parameters");
                goto fail;
            }
        }
        nd = n;
        memcpy(shape, ss, sizeof(int64_t) * (size_t)n);
    }
    {
        zval res;
        if (tsr_ndarray_new(&res, D_F64_ID, nd, shape, 0) == FAILURE) goto fail;
        void *ptrs[17];
        int64_t strides[17 * 32];
        for (int k = 0; k < nops; k++) {
            const tsr_array *a = &Z_TSR(ops[k])->a;
            ptrs[k] = (char *)a->data + a->offset;
            if (nd > 0) tsr_broadcast_strides(a, nd, shape, strides + (size_t)k * nd);
        }
        const tsr_array *o = &Z_TSR(res)->a;
        ptrs[nops] = (char *)o->data + o->offset;
        for (int32_t k = 0; k < nd; k++) strides[(size_t)nops * nd + k] = o->strides[k];
        const int rc = tsr_fn_rvs(d->id, tsr_generator_state(rng), nd, nd ? shape : NULL, nops + 1, ptrs, nd ? strides : NULL);
        for (int k = 0; k < nops; k++) zval_ptr_dtor(&ops[k]);
        nops = 0;
        if (Z_TYPE(gen) != IS_UNDEF) zval_ptr_dtor(&gen);
        if (rc < 0) {
            zval_ptr_dtor(&res);
            zend_value_error("%s()->rvs(): %s", d->label, tsr_fn_error());
            RETURN_THROWS();
        }
        char info[256];
        const int discrete = tsr_fn_info(d->id, info, sizeof info) > 0 && strstr(info, "\"discrete\":true") != NULL;
        if (nd == 0) {
            const double v = *(double *)ptrs[nops];
            zval_ptr_dtor(&res);
            if (discrete) RETURN_LONG((zend_long)v);
            RETURN_DOUBLE(v);
        }
        if (discrete) {
            zval ints;
            if (tsr_ndarray_contiguous_as(&res, D_I64_ID, &ints) == FAILURE) { zval_ptr_dtor(&res); RETURN_THROWS(); }
            zval_ptr_dtor(&res);
            RETURN_COPY_VALUE(&ints);
        }
        RETURN_COPY_VALUE(&res);
    }
fail:
    for (int k = 0; k < nops; k++) zval_ptr_dtor(&ops[k]);
    if (Z_TYPE(gen) != IS_UNDEF) zval_ptr_dtor(&gen);
}

static PHP_METHOD(Distribution, entropy) { ZEND_PARSE_PARAMETERS_NONE(); call_method(ZEND_THIS, DM_ENTROPY, NULL, return_value); }

static PHP_METHOD(Distribution, support)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zval r;
    call_method(ZEND_THIS, DM_SUPPORT, NULL, &r);
    if (EG(exception)) RETURN_THROWS();
    array_init_size(return_value, 2);
    zval *a = zend_hash_str_find(Z_ARRVAL(r), "a", 1), *b = zend_hash_str_find(Z_ARRVAL(r), "b", 1);
    Z_TRY_ADDREF_P(a);
    Z_TRY_ADDREF_P(b);
    add_next_index_zval(return_value, a);
    add_next_index_zval(return_value, b);
    zval_ptr_dtor(&r);
}

static PHP_METHOD(Distribution, median)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zval half;
    ZVAL_DOUBLE(&half, 0.5);
    call_method(ZEND_THIS, DM_PPF, &half, return_value);
}

static PHP_METHOD(Distribution, interval)
{
    double c;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_DOUBLE(c)
    ZEND_PARSE_PARAMETERS_END();
    if (!(c >= 0 && c <= 1)) { zend_argument_value_error(1, "must be between 0 and 1"); RETURN_THROWS(); }
    zval lo, hi, a, b;
    ZVAL_DOUBLE(&lo, (1.0 - c) / 2.0);
    ZVAL_DOUBLE(&hi, (1.0 + c) / 2.0);
    call_method(ZEND_THIS, DM_PPF, &lo, &a);
    if (EG(exception)) RETURN_THROWS();
    call_method(ZEND_THIS, DM_PPF, &hi, &b);
    if (EG(exception)) { zval_ptr_dtor(&a); RETURN_THROWS(); }
    array_init_size(return_value, 2);
    add_next_index_zval(return_value, &a);
    add_next_index_zval(return_value, &b);
}

static PHP_METHOD(Distribution, name)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_dist *d = Z_DIST_P(ZEND_THIS);
    if (d->id == PB_ID) RETURN_STRING("poisson_binom");
    RETURN_STRING(tsr_fn_name(d->id) + 6);
}

static PHP_METHOD(Distribution, args)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_dist *d = Z_DIST_P(ZEND_THIS);
    array_init_size(return_value, (uint32_t)d->nparams);
    for (int k = 0; k < d->nparams; k++) { Z_TRY_ADDREF(d->params[k]); add_next_index_zval(return_value, &d->params[k]); }
}

ZEND_BEGIN_ARG_INFO_EX(ai_dist_x, 0, 0, 1)
    ZEND_ARG_INFO(0, x)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_dist_q, 0, 0, 1)
    ZEND_ARG_INFO(0, q)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_dist_none, 0, 0, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_dist_stats, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, moments, IS_STRING, 0, "'mv'")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_dist_moment, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, order, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_dist_rvs, 0, 0, 0)
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, size, "null")
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, randomState, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_dist_interval, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, confidence, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()

static const zend_function_entry dist_methods[] = {
    PHP_ME(Distribution, pdf, ai_dist_x, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, logpdf, ai_dist_x, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, pmf, ai_dist_x, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, logpmf, ai_dist_x, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, cdf, ai_dist_x, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, logcdf, ai_dist_x, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, sf, ai_dist_x, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, logsf, ai_dist_x, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, ppf, ai_dist_q, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, isf, ai_dist_q, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, stats, ai_dist_stats, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, mean, ai_dist_none, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, var, ai_dist_none, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, std, ai_dist_none, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, entropy, ai_dist_none, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, moment, ai_dist_moment, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, rvs, ai_dist_rvs, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, support, ai_dist_none, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, median, ai_dist_none, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, interval, ai_dist_interval, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, name, ai_dist_none, ZEND_ACC_PUBLIC)
    PHP_ME(Distribution, args, ai_dist_none, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

void tsr_register_dist(void)
{
    zend_class_entry ce;
    INIT_NS_CLASS_ENTRY(ce, "Tessero\\Ext", "Distribution", dist_methods);
    tsr_ce_distribution = zend_register_internal_class(&ce);
    tsr_ce_distribution->ce_flags |= ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES | ZEND_ACC_NOT_SERIALIZABLE;
    tsr_ce_distribution->create_object = dist_create;
    memcpy(&dist_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    dist_handlers.offset = XtOffsetOf(tsr_dist, std);
    dist_handlers.free_obj = dist_free;
    dist_handlers.clone_obj = NULL;
}
