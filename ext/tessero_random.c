/*
 * Tessero\Ext\Random\Generator: numpy.random.Generator over the kernel's PCG64 (NumPy's stream bit for bit).
 *
 *   $g = Generator::defaultRng(42);            // numpy.random.default_rng(42)
 *   $g->gamma(2.0, size: 5);                   // the distribution methods come from the kernel's function
 *   $g->integers(0, 10, size: 3);              // registry (kind "random", tessero_fn.c); the rest are here
 *
 * The object holds the 6-word PCG64 state (words 4-5: numpy's buffered 32-bit half).
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#if PHP_VERSION_ID >= 80200
#include "ext/random/php_random.h"
#else
#include "ext/standard/php_random.h"
#endif
#include "Zend/zend_exceptions.h"
#include "libtessero/src/internal.h"
#define TSR_TESSERO_H_INCLUDED
#include "php_tessero.h"

zend_class_entry *tsr_ce_generator;
static zend_object_handlers generator_handlers;

typedef struct {
    uint64_t st[6];
    zend_object std;
} tsr_gen;

static inline tsr_gen *gen_of(zend_object *o) { return (tsr_gen *)((char *)o - XtOffsetOf(tsr_gen, std)); }

uint64_t *tsr_generator_state(zval *obj) { return gen_of(Z_OBJ_P(obj))->st; }

static zend_object *gen_create(zend_class_entry *ce)
{
    tsr_gen *g = zend_object_alloc(sizeof(tsr_gen), ce);
    memset(g->st, 0, sizeof g->st);
    zend_object_std_init(&g->std, ce);
    object_properties_init(&g->std, ce);
    g->std.handlers = &generator_handlers;
    return &g->std;
}

static zend_object *gen_clone(zend_object *old)
{
    zend_object *n = gen_create(old->ce);
    memcpy(gen_of(n)->st, gen_of(old)->st, sizeof gen_of(n)->st);
    zend_objects_clone_members(n, old);
    return n;
}

/* SeedSequence(seed) -> PCG64, as numpy.random.default_rng does: an int becomes little-endian 32-bit words */
static int seed_words(zval *seed, uint32_t *words, int *n)
{
    *n = 0;
    if (seed == NULL || Z_TYPE_P(seed) == IS_NULL) {
        for (int k = 0; k < 4; k++) {
            uint32_t w;
            if (php_random_bytes_throw(&w, sizeof w) == FAILURE) return FAILURE;
            words[(*n)++] = w;
        }
        return SUCCESS;
    }
    if (Z_TYPE_P(seed) == IS_LONG) {
        const zend_long s = Z_LVAL_P(seed);
        if (s < 0) { zend_argument_value_error(1, "must be a non-negative integer"); return FAILURE; }
        words[(*n)++] = (uint32_t)((uint64_t)s & 0xffffffffu);
        if ((uint64_t)s > 0xffffffffu) words[(*n)++] = (uint32_t)((uint64_t)s >> 32);
        return SUCCESS;
    }
    if (Z_TYPE_P(seed) == IS_ARRAY) {
        zval *e;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(seed), e) {
            if (Z_TYPE_P(e) != IS_LONG || Z_LVAL_P(e) < 0 || Z_LVAL_P(e) > 0xffffffffL || *n >= 64) {
                zend_argument_value_error(1, "must be a list of at most 64 integers in [0, 2^32)");
                return FAILURE;
            }
            words[(*n)++] = (uint32_t)Z_LVAL_P(e);
        } ZEND_HASH_FOREACH_END();
        return SUCCESS;
    }
    zend_argument_type_error(1, "must be an int, a list of ints or null");
    return FAILURE;
}

static PHP_METHOD(Generator, defaultRng)
{
    zval *seed = NULL;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(seed)
    ZEND_PARSE_PARAMETERS_END();
    uint32_t words[64];
    int n;
    if (seed_words(seed, words, &n) == FAILURE) RETURN_THROWS();
    uint64_t pool[4];
    tsr_seed_sequence(words, n, pool, 4);
    object_init_ex(return_value, tsr_ce_generator);
    tsr_pcg64_seed(gen_of(Z_OBJ_P(return_value))->st, pool[0], pool[1], pool[2], pool[3]);
}

/* size argument: null, int or list of ints -> shape; returns -1 for null */
static int size_shape(zval *size, int64_t *shape)
{
    if (size == NULL || Z_TYPE_P(size) == IS_NULL) return -1;
    if (Z_TYPE_P(size) == IS_LONG) {
        if (Z_LVAL_P(size) < 0) { zend_value_error("negative dimensions are not allowed"); return -2; }
        shape[0] = Z_LVAL_P(size);
        return 1;
    }
    if (Z_TYPE_P(size) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_P(size)) > 32) {
        zend_type_error("size must be an int, a list of ints or null");
        return -2;
    }
    int nd = 0;
    zval *e;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(size), e) {
        if (Z_TYPE_P(e) != IS_LONG || Z_LVAL_P(e) < 0) { zend_value_error("negative dimensions are not allowed"); return -2; }
        shape[nd++] = Z_LVAL_P(e);
    } ZEND_HASH_FOREACH_END();
    return nd;
}

/* integers(low, high = null, size = null, endpoint = false): numpy's bounded int64 stream */
static PHP_METHOD(Generator, integers)
{
    zend_long low, high = 0;
    zend_bool high_null = 1, endpoint = 0;
    zval *size = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_LONG(low)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG_OR_NULL(high, high_null)
        Z_PARAM_ZVAL_OR_NULL(size)
        Z_PARAM_BOOL(endpoint)
    ZEND_PARSE_PARAMETERS_END();
    if (high_null) { high = low; low = 0; }
    if (endpoint) {
        if (high == ZEND_LONG_MAX) { zend_argument_value_error(2, "is too large for endpoint=true"); RETURN_THROWS(); }
        high++;
    }
    if (high <= low) { zend_argument_value_error(2, "must be greater than low (low >= high)"); RETURN_THROWS(); }
    int64_t shape[32];
    const int nd = size_shape(size, shape);
    if (nd == -2) RETURN_THROWS();
    uint64_t *st = gen_of(Z_OBJ_P(ZEND_THIS))->st;
    if (nd < 0) {
        int64_t v;
        tsr_pcg64_integers(st, 1, low, high, &v);
        RETURN_LONG((zend_long)v);
    }
    if (tsr_ndarray_new(return_value, 2 /* int64 */, nd, shape, 0) == FAILURE) RETURN_THROWS();
    const tsr_array *a = &Z_TSR_P(return_value)->a;
    const int64_t n = tsr_array_size(a);
    if (n > 0) tsr_pcg64_integers(st, n, low, high, (int64_t *)((char *)a->data + a->offset));
}

/* shuffle(x): in place along axis 0 (Generator.shuffle) */
static PHP_METHOD(Generator, shuffle)
{
    zval *x;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(x, tsr_ce_ndarray)
    ZEND_PARSE_PARAMETERS_END();
    tsr_array *a = &Z_TSR_P(x)->a;
    if (a->ndim == 0) { zend_throw_exception(tsr_ce_shape_exception, "shuffle needs at least one dimension", 0); RETURN_THROWS(); }
    if (!tsr_array_is_contiguous(a)) { zend_throw_exception(tsr_ce_exception, "shuffle needs a C-contiguous array", 0); RETURN_THROWS(); }
    const int64_t n = a->shape[0];
    const int64_t item = n ? tsr_array_size(a) / n * tsr_itemsize(a->dtype) : 0;
    if (n > 1 && tsr_pcg64_shuffle(gen_of(Z_OBJ_P(ZEND_THIS))->st, n, item, (char *)a->data + a->offset) != 0) {
        zend_throw_exception(tsr_ce_memory_exception, "shuffle: out of memory", 0);
        RETURN_THROWS();
    }
}

/* permutation(x): a shuffled copy (an int n: a permutation of arange(n)) */
static PHP_METHOD(Generator, permutation)
{
    zval *x;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(x)
    ZEND_PARSE_PARAMETERS_END();
    uint64_t *st = gen_of(Z_OBJ_P(ZEND_THIS))->st;
    if (Z_TYPE_P(x) == IS_LONG) {
        if (Z_LVAL_P(x) < 0) { zend_argument_value_error(1, "must be non-negative"); RETURN_THROWS(); }
        int64_t shape[1] = {Z_LVAL_P(x)};
        if (tsr_ndarray_new(return_value, 2, 1, shape, 0) == FAILURE) RETURN_THROWS();
        int64_t *p = (int64_t *)Z_TSR_P(return_value)->a.data;
        for (int64_t i = 0; i < shape[0]; i++) p[i] = i;
        if (shape[0] > 1) tsr_pcg64_shuffle(st, shape[0], 8, p);
        return;
    }
    zval src;
    if (tsr_ndarray_from_zval(x, -1, &src) == FAILURE) RETURN_THROWS();
    const tsr_array *s = &Z_TSR(src)->a;
    if (s->ndim == 0) { zval_ptr_dtor(&src); zend_throw_exception(tsr_ce_shape_exception, "permutation needs at least one dimension", 0); RETURN_THROWS(); }
    if (tsr_ndarray_contiguous_as(&src, s->dtype, return_value) == FAILURE) { zval_ptr_dtor(&src); RETURN_THROWS(); }
    /* contiguous_as may hand back the source itself: shuffle a private copy */
    if (Z_OBJ_P(return_value) == Z_OBJ(src) || Z_TSR_P(return_value)->a.data == s->data) {
        zval c;
        const tsr_array *r = &Z_TSR_P(return_value)->a;
        if (tsr_ndarray_new(&c, r->dtype, r->ndim, r->shape, 0) == FAILURE) { zval_ptr_dtor(&src); RETURN_THROWS(); }
        memcpy(Z_TSR(c)->a.data, (char *)r->data + r->offset, (size_t)(tsr_array_size(r) * tsr_itemsize(r->dtype)));
        zval_ptr_dtor(return_value);
        ZVAL_COPY_VALUE(return_value, &c);
    }
    zval_ptr_dtor(&src);
    tsr_array *a = &Z_TSR_P(return_value)->a;
    const int64_t n = a->shape[0];
    const int64_t item = n ? tsr_array_size(a) / n * tsr_itemsize(a->dtype) : 0;
    if (n > 1) tsr_pcg64_shuffle(st, n, item, (char *)a->data + a->offset);
}

static PHP_METHOD(Generator, getState)
{
    ZEND_PARSE_PARAMETERS_NONE();
    const uint64_t *st = gen_of(Z_OBJ_P(ZEND_THIS))->st;
    array_init_size(return_value, 6);
    for (int k = 0; k < 6; k++) add_next_index_long(return_value, (zend_long)st[k]);
}

static PHP_METHOD(Generator, setState)
{
    HashTable *words;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(words)
    ZEND_PARSE_PARAMETERS_END();
    if (zend_hash_num_elements(words) != 6) { zend_argument_value_error(1, "must hold the 6 state words of getState()"); RETURN_THROWS(); }
    uint64_t st[6];
    int k = 0;
    zval *e;
    ZEND_HASH_FOREACH_VAL(words, e) {
        if (Z_TYPE_P(e) != IS_LONG) { zend_argument_type_error(1, "must hold integers"); RETURN_THROWS(); }
        st[k++] = (uint64_t)Z_LVAL_P(e);
    } ZEND_HASH_FOREACH_END();
    memcpy(gen_of(Z_OBJ_P(ZEND_THIS))->st, st, sizeof st);
}

/* searchsorted(cdf, x, side='right'): first index with cdf[idx] > x. */
static int64_t choice_ss_right(const double *cdf, int64_t m, double x)
{
    int64_t lo = 0, hi = m;
    while (lo < hi) { int64_t mid = (lo + hi) >> 1; if (cdf[mid] <= x) lo = mid + 1; else hi = mid; }
    return lo;
}

static void choice_elem_to_zval(int dt, const char *p, zval *z)
{
    switch (dt) {
    case TSR_F64: ZVAL_DOUBLE(z, *(const double *)p); break;
    case TSR_F32: ZVAL_DOUBLE(z, (double)*(const float *)p); break;
    case TSR_I64: ZVAL_LONG(z, (zend_long)*(const int64_t *)p); break;
    case TSR_I32: ZVAL_LONG(z, *(const int32_t *)p); break;
    case TSR_U8:  ZVAL_LONG(z, *(const uint8_t *)p); break;
    case TSR_BOOL: ZVAL_BOOL(z, *(const uint8_t *)p != 0); break;
    case TSR_C128:
        array_init_size(z, 2);
        add_next_index_double(z, ((const double *)p)[0]);
        add_next_index_double(z, ((const double *)p)[1]);
        break;
    default: ZVAL_NULL(z);
    }
}

/* choice(a, size=null, replace=true, p=null): numpy.random.Generator.choice over a 1-D population.
   Uses the same draws as the FFI method (integers / permutation / cdf.searchsorted on random()). */
static PHP_METHOD(Generator, choice)
{
    zval *a, *size = NULL, *p = NULL;
    zend_bool replace = 1;
    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_ZVAL(a)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(size)
        Z_PARAM_BOOL(replace)
        Z_PARAM_ARRAY_OR_NULL(p)
    ZEND_PARSE_PARAMETERS_END();
    uint64_t *st = gen_of(Z_OBJ_P(ZEND_THIS))->st;

    zval pop;
    int64_t n;
    if (Z_TYPE_P(a) == IS_LONG) {
        if (Z_LVAL_P(a) <= 0) { zend_argument_value_error(1, "must be greater than 0"); RETURN_THROWS(); }
        n = Z_LVAL_P(a);
        int64_t sh[1] = { n };
        if (tsr_ndarray_new(&pop, 2, 1, sh, 0) == FAILURE) RETURN_THROWS();
        int64_t *pp = (int64_t *)((char *)Z_TSR(pop)->a.data + Z_TSR(pop)->a.offset);
        for (int64_t i = 0; i < n; i++) pp[i] = i;
    } else {
        zval src;
        if (tsr_ndarray_from_zval(a, -1, &src) == FAILURE) RETURN_THROWS();
        if (Z_TSR(src)->a.ndim != 1) { zval_ptr_dtor(&src); zend_throw_exception(tsr_ce_shape_exception, "choice: population must be 1-D.", 0); RETURN_THROWS(); }
        if (tsr_ndarray_contiguous_as(&src, Z_TSR(src)->a.dtype, &pop) == FAILURE) { zval_ptr_dtor(&src); RETURN_THROWS(); }
        zval_ptr_dtor(&src);
        n = Z_TSR(pop)->a.shape[0];
        if (n == 0) { zval_ptr_dtor(&pop); zend_throw_exception(tsr_ce_exception, "choice: population is empty.", 0); RETURN_THROWS(); }
    }
    const int dt = Z_TSR(pop)->a.dtype;
    const int64_t isz = tsr_itemsize(dt);
    const char *base = (const char *)Z_TSR(pop)->a.data + Z_TSR(pop)->a.offset;

    int64_t shape[32];
    const int nd = size_shape(size, shape);
    if (nd == -2) { zval_ptr_dtor(&pop); RETURN_THROWS(); }
    const int scalar = (nd < 0);
    int64_t count = 1;
    for (int k = 0; k < nd; k++) count *= shape[k];
    if (scalar) count = 1;

    int64_t *idx = (int64_t *)emalloc((size_t)(count > 0 ? count : 1) * sizeof(int64_t));
    if (p != NULL) {
        if (!replace) { efree(idx); zval_ptr_dtor(&pop); zend_throw_exception(tsr_ce_exception, "choice without replacement and with p is not supported.", 0); RETURN_THROWS(); }
        if ((int64_t)zend_hash_num_elements(Z_ARRVAL_P(p)) != n) { efree(idx); zval_ptr_dtor(&pop); zend_throw_exception(tsr_ce_shape_exception, "choice: p must have one probability per element.", 0); RETURN_THROWS(); }
        double *cdf = (double *)emalloc((size_t)n * sizeof(double));
        double acc = 0; int64_t i = 0;
        zval *e;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(p), e) { acc += zval_get_double(e); cdf[i++] = acc; } ZEND_HASH_FOREACH_END();
        const double tot = cdf[n - 1];
        for (int64_t k = 0; k < n; k++) cdf[k] /= tot;
        double *u = (double *)emalloc((size_t)(count > 0 ? count : 1) * sizeof(double));
        tsr_pcg64_random(st, count, u);
        for (int64_t k = 0; k < count; k++) idx[k] = choice_ss_right(cdf, n, u[k]);
        efree(u); efree(cdf);
    } else if (replace) {
        tsr_pcg64_integers(st, count, 0, n, idx);
    } else {
        if (count > n) { efree(idx); zval_ptr_dtor(&pop); zend_throw_exception(tsr_ce_exception, "choice: cannot take a larger sample than the population when replace=false.", 0); RETURN_THROWS(); }
        int64_t *perm = (int64_t *)emalloc((size_t)n * sizeof(int64_t));
        for (int64_t k = 0; k < n; k++) perm[k] = k;
        if (n > 1) tsr_pcg64_shuffle(st, n, 8, perm);
        for (int64_t k = 0; k < count; k++) idx[k] = perm[k];
        efree(perm);
    }

    if (scalar) {
        choice_elem_to_zval(dt, base + idx[0] * isz, return_value);
        efree(idx); zval_ptr_dtor(&pop);
        return;
    }
    if (tsr_ndarray_new(return_value, dt, nd, shape, 0) == FAILURE) { efree(idx); zval_ptr_dtor(&pop); RETURN_THROWS(); }
    char *out = (char *)Z_TSR_P(return_value)->a.data + Z_TSR_P(return_value)->a.offset;
    for (int64_t k = 0; k < count; k++) memcpy(out + k * isz, base + idx[k] * isz, (size_t)isz);
    efree(idx);
    zval_ptr_dtor(&pop);
}

/* dirichlet(alpha, size=null): numpy.random.Generator.dirichlet via the standard-gamma method -
   draw one standard_gamma per component (C order, component-minor) then normalise each sample. */
static PHP_METHOD(Generator, dirichlet)
{
    zval *alpha_z, *size = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY(alpha_z)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(size)
    ZEND_PARSE_PARAMETERS_END();
    uint64_t *st = gen_of(Z_OBJ_P(ZEND_THIS))->st;

    const int64_t K = zend_hash_num_elements(Z_ARRVAL_P(alpha_z));
    double *alpha = (double *)emalloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    {
        int64_t i = 0; zval *e;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(alpha_z), e) {
            double a = zval_get_double(e);
            if (a < 0) { efree(alpha); zend_throw_exception(tsr_ce_exception, "dirichlet: alpha must be non-negative.", 0); RETURN_THROWS(); }
            alpha[i++] = a;
        } ZEND_HASH_FOREACH_END();
    }

    int64_t sdims[32];
    const int snd = size_shape(size, sdims);
    if (snd == -2) { efree(alpha); RETURN_THROWS(); }
    int nd;
    int64_t oshape[40];
    if (snd < 0) { nd = 1; oshape[0] = K; }
    else { nd = snd + 1; for (int d = 0; d < snd; d++) oshape[d] = sdims[d]; oshape[snd] = K; }

    zval av;
    { int64_t ash[1] = { K };
      if (tsr_ndarray_new(&av, TSR_F64, 1, ash, 0) == FAILURE) { efree(alpha); RETURN_THROWS(); }
      memcpy((char *)Z_TSR(av)->a.data + Z_TSR(av)->a.offset, alpha, (size_t)K * sizeof(double)); }
    efree(alpha);

    if (tsr_ndarray_new(return_value, TSR_F64, nd, oshape, 0) == FAILURE) { zval_ptr_dtor(&av); RETURN_THROWS(); }
    tsr_array *o = &Z_TSR_P(return_value)->a;

    const int gid = tsr_fn_find("random.standard_gamma");
    if (gid < 0) { zval_ptr_dtor(&av); zval_ptr_dtor(return_value); zend_throw_error(NULL, "random.standard_gamma not registered"); RETURN_THROWS(); }
    double pv[1] = { 0.0 };                 /* dtype=float64 */
    void *ptrs[2];
    int64_t strides[2 * 40];
    const tsr_array *aarr = &Z_TSR(av)->a;
    ptrs[0] = (char *)aarr->data + aarr->offset;
    tsr_broadcast_strides(aarr, nd, oshape, strides);
    ptrs[1] = (char *)o->data + o->offset;
    for (int d = 0; d < nd; d++) strides[nd + d] = o->strides[d];
    const int64_t total = tsr_array_size(o);
    int rc = total > 0 ? tsr_fn_random(gid, st, pv, nd, oshape, 2, ptrs, strides, TSR_F64) : 0;
    zval_ptr_dtor(&av);
    if (rc < 0) { zval_ptr_dtor(return_value); zend_throw_exception(tsr_ce_exception, "dirichlet: sampling failed", 0); RETURN_THROWS(); }

    double *out = (double *)((char *)o->data + o->offset);
    for (int64_t r = 0; K > 0 && r + K <= total; r += K) {
        double sum = 0;
        for (int64_t j = 0; j < K; j++) sum += out[r + j];
        const double inv = 1.0 / sum;
        for (int64_t j = 0; j < K; j++) out[r + j] *= inv;
    }
}

/* multinomial(n, pvals, size=null): sequential conditional binomials, matching numpy. */
static PHP_METHOD(Generator, multinomial)
{
    zend_long n;
    zval *pvals_z, *size = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_LONG(n)
        Z_PARAM_ARRAY(pvals_z)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(size)
    ZEND_PARSE_PARAMETERS_END();
    uint64_t *st = gen_of(Z_OBJ_P(ZEND_THIS))->st;

    const int64_t K = zend_hash_num_elements(Z_ARRVAL_P(pvals_z));
    if (K < 1) { zend_throw_exception(tsr_ce_exception, "multinomial: pvals must have at least one element.", 0); RETURN_THROWS(); }
    double *pv = (double *)emalloc((size_t)K * sizeof(double));
    { int64_t i = 0; zval *e; ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(pvals_z), e) { pv[i++] = zval_get_double(e); } ZEND_HASH_FOREACH_END(); }
    double head = 0.0;
    for (int64_t j = 0; j < K - 1; j++) head += pv[j];
    if (head > 1.0 + 1e-12) { efree(pv); zend_throw_exception(tsr_ce_exception, "multinomial: sum(pvals[:-1]) > 1.0", 0); RETURN_THROWS(); }

    int64_t sdims[32];
    const int snd = size_shape(size, sdims);
    if (snd == -2) { efree(pv); RETURN_THROWS(); }
    int nd;
    int64_t oshape[40], rows = 1;
    if (snd < 0) { nd = 1; oshape[0] = K; }
    else { nd = snd + 1; for (int d = 0; d < snd; d++) { oshape[d] = sdims[d]; rows *= sdims[d]; } oshape[snd] = K; }
    if (snd < 0) rows = 1;
    if (snd >= 0 && snd == 0) rows = 1;

    if (tsr_ndarray_new(return_value, TSR_I64, nd, oshape, 0) == FAILURE) { efree(pv); RETURN_THROWS(); }
    tsr_array *o = &Z_TSR_P(return_value)->a;
    int64_t *out = (int64_t *)((char *)o->data + o->offset);

    const int bid = tsr_fn_find("random.binomial");
    if (bid < 0) { efree(pv); zval_ptr_dtor(return_value); zend_throw_error(NULL, "random.binomial not registered"); RETURN_THROWS(); }
    double dummy[1] = { 0.0 };
    for (int64_t r = 0; r < rows; r++) {
        int64_t remaining = n;
        double psum = 1.0;
        for (int64_t j = 0; j < K - 1; j++) {
            double pj = psum > 0.0 ? pv[j] / psum : 0.0;
            if (pj > 1.0) pj = 1.0;
            int64_t bj = 0;
            if (remaining > 0) {
                double nd_in = (double)remaining, pj_in = pj;
                int64_t res_i = 0;
                void *ptrs[3] = { &nd_in, &pj_in, &res_i };
                int rc = tsr_fn_random(bid, st, dummy, 0, NULL, 3, ptrs, NULL, TSR_I64);
                if (rc < 0) { efree(pv); zval_ptr_dtor(return_value); zend_throw_exception(tsr_ce_exception, "multinomial: sampling failed", 0); RETURN_THROWS(); }
                bj = res_i;
            }
            out[r * K + j] = bj;
            remaining -= bj;
            psum -= pv[j];
        }
        out[r * K + (K - 1)] = remaining;
    }
    efree(pv);
}

/* permuted(x, axis=null, out=null): per-slice shuffle along axis (numpy.random.Generator.permuted).
   Works on a private contiguous copy; axis=null permutes the flattened array. */
static PHP_METHOD(Generator, permuted)
{
    zval *x, *axis = NULL, *out = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_OBJECT_OF_CLASS(x, tsr_ce_ndarray)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(axis)
        Z_PARAM_ZVAL_OR_NULL(out)
    ZEND_PARSE_PARAMETERS_END();
    uint64_t *st = gen_of(Z_OBJ_P(ZEND_THIS))->st;

    const tsr_array *xa = &Z_TSR_P(x)->a;
    const int nd = xa->ndim;
    const int isz = (int)tsr_itemsize(xa->dtype);
    if (tsr_ndarray_contiguous_as(x, xa->dtype, return_value) == FAILURE) RETURN_THROWS();
    if (Z_OBJ_P(return_value) == Z_OBJ_P(x) || Z_TSR_P(return_value)->a.data == xa->data) {
        zval c;
        const tsr_array *r = &Z_TSR_P(return_value)->a;
        if (tsr_ndarray_new(&c, r->dtype, r->ndim, r->shape, 0) == FAILURE) RETURN_THROWS();
        memcpy((char *)Z_TSR(c)->a.data, (char *)r->data + r->offset, (size_t)(tsr_array_size(r) * isz));
        zval_ptr_dtor(return_value);
        ZVAL_COPY_VALUE(return_value, &c);
    }
    tsr_array *o = &Z_TSR_P(return_value)->a;
    char *data = (char *)o->data + o->offset;
    const int64_t total = tsr_array_size(o);

    if (axis == NULL || Z_TYPE_P(axis) == IS_NULL) {
        if (total > 1) tsr_pcg64_shuffle(st, total, isz, data);
    } else {
        if (Z_TYPE_P(axis) != IS_LONG) { zval_ptr_dtor(return_value); zend_type_error("permuted(): axis must be an integer or null"); RETURN_THROWS(); }
        int64_t k = Z_LVAL_P(axis);
        if (k < 0) k += nd;
        if (k < 0 || k >= nd) { zval_ptr_dtor(return_value); zend_argument_value_error(2, "is out of bounds for the array"); RETURN_THROWS(); }
        const int64_t L = o->shape[k];
        int64_t inner = 1, outer = 1;
        for (int d = (int)k + 1; d < nd; d++) inner *= o->shape[d];
        for (int d = 0; d < (int)k; d++) outer *= o->shape[d];
        if (L > 1) {
            char *tmp = (char *)emalloc((size_t)L * isz);
            for (int64_t ob = 0; ob < outer; ob++) {
                for (int64_t ib = 0; ib < inner; ib++) {
                    char *base = data + (size_t)(ob * L * inner + ib) * isz;
                    for (int64_t j = 0; j < L; j++) memcpy(tmp + (size_t)j * isz, base + (size_t)(j * inner) * isz, (size_t)isz);
                    tsr_pcg64_shuffle(st, L, isz, tmp);
                    for (int64_t j = 0; j < L; j++) memcpy(base + (size_t)(j * inner) * isz, tmp + (size_t)j * isz, (size_t)isz);
                }
            }
            efree(tmp);
        }
    }

    if (out != NULL && Z_TYPE_P(out) == IS_OBJECT) {
        tsr_array *oa = &Z_TSR_P(out)->a;
        if (tsr_array_size(oa) == total && tsr_array_is_contiguous(oa)) {
            memcpy((char *)oa->data + oa->offset, data, (size_t)(total * isz));
            zval_ptr_dtor(return_value);
            ZVAL_COPY(return_value, out);
        }
    }
}

/* multivariateHypergeometric(colors, nsample, size=null, method='marginals'): sequential hypergeometric draws. */
static PHP_METHOD(Generator, multivariateHypergeometric)
{
    zval *colors_z, *size = NULL;
    zend_long nsample;
    zend_string *method = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_ARRAY(colors_z)
        Z_PARAM_LONG(nsample)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(size)
        Z_PARAM_STR_OR_NULL(method)
    ZEND_PARSE_PARAMETERS_END();
    uint64_t *st = gen_of(Z_OBJ_P(ZEND_THIS))->st;
    if (method != NULL && strcmp(ZSTR_VAL(method), "marginals") != 0) {
        zend_throw_exception(tsr_ce_exception, "multivariate_hypergeometric: only method='marginals' is supported.", 0); RETURN_THROWS();
    }

    const int64_t K = zend_hash_num_elements(Z_ARRVAL_P(colors_z));
    if (K < 1) { zend_throw_exception(tsr_ce_exception, "multivariate_hypergeometric: colors must have at least one element.", 0); RETURN_THROWS(); }
    int64_t *colors = (int64_t *)emalloc((size_t)K * sizeof(int64_t));
    int64_t total = 0;
    { int64_t i = 0; zval *e; ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(colors_z), e) {
        int64_t c = zval_get_long(e);
        if (c < 0) { efree(colors); zend_throw_exception(tsr_ce_exception, "multivariate_hypergeometric: colors must be nonnegative.", 0); RETURN_THROWS(); }
        colors[i++] = c; total += c;
    } ZEND_HASH_FOREACH_END(); }
    if (nsample < 0) { efree(colors); zend_throw_exception(tsr_ce_exception, "multivariate_hypergeometric: nsample must be nonnegative.", 0); RETURN_THROWS(); }
    if (nsample > total) { efree(colors); zend_throw_exception(tsr_ce_exception, "multivariate_hypergeometric: nsample > sum(colors)", 0); RETURN_THROWS(); }

    int64_t sdims[32];
    const int snd = size_shape(size, sdims);
    if (snd == -2) { efree(colors); RETURN_THROWS(); }
    int nd;
    int64_t oshape[40], rows = 1;
    if (snd < 0) { nd = 1; oshape[0] = K; }
    else { nd = snd + 1; for (int d = 0; d < snd; d++) { oshape[d] = sdims[d]; rows *= sdims[d]; } oshape[snd] = K; }

    if (tsr_ndarray_new(return_value, TSR_I64, nd, oshape, 0) == FAILURE) { efree(colors); RETURN_THROWS(); }
    tsr_array *o = &Z_TSR_P(return_value)->a;
    int64_t *out = (int64_t *)((char *)o->data + o->offset);

    const int hid = tsr_fn_find("random.hypergeometric");
    if (hid < 0) { efree(colors); zval_ptr_dtor(return_value); zend_throw_error(NULL, "random.hypergeometric not registered"); RETURN_THROWS(); }
    double dummy[1] = { 0.0 };
    for (int64_t r = 0; r < rows; r++) {
        int64_t remaining = total, ns = nsample;
        for (int64_t j = 0; j < K - 1; j++) {
            int64_t rv = 0;
            if (ns > 0 && remaining > 0) {
                double ng = (double)colors[j], nb = (double)(remaining - colors[j]), nns = (double)ns;
                int64_t res_i = 0;
                void *ptrs[4] = { &ng, &nb, &nns, &res_i };
                int rc = tsr_fn_random(hid, st, dummy, 0, NULL, 4, ptrs, NULL, TSR_I64);
                if (rc < 0) { efree(colors); zval_ptr_dtor(return_value); zend_throw_exception(tsr_ce_exception, "multivariate_hypergeometric: sampling failed", 0); RETURN_THROWS(); }
                rv = res_i;
            }
            out[r * K + j] = rv;
            ns -= rv;
            remaining -= colors[j];
        }
        out[r * K + (K - 1)] = ns;
    }
    efree(colors);
}

/* bytes(length): `length` random bytes (numpy.random.Generator.bytes). length 0 still advances the stream. */
static PHP_METHOD(Generator, bytes)
{
    zend_long length;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(length)
    ZEND_PARSE_PARAMETERS_END();
    if (length < 0) { zend_argument_value_error(1, "must be non-negative"); RETURN_THROWS(); }
    zend_string *out = zend_string_alloc((size_t)length, 0);
    tsr_pcg64_bytes(gen_of(Z_OBJ_P(ZEND_THIS))->st, length, (uint8_t *)ZSTR_VAL(out));
    ZSTR_VAL(out)[length] = '\0';
    RETURN_STR(out);
}

/* multivariate_normal(mean, cov, size=null, check_valid='warn', tol=1e-8, method='svd'): SVD-factorised draws. */
static PHP_METHOD(Generator, multivariateNormal)
{
    zval *mean_z, *cov_z, *size = NULL, *cv = NULL, *tl = NULL;
    zend_string *method = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 6)
        Z_PARAM_ARRAY(mean_z)
        Z_PARAM_ARRAY(cov_z)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(size)
        Z_PARAM_ZVAL_OR_NULL(cv)
        Z_PARAM_ZVAL_OR_NULL(tl)
        Z_PARAM_STR_OR_NULL(method)
    ZEND_PARSE_PARAMETERS_END();
    (void)cv; (void)tl;
    if (method && !zend_string_equals_literal(method, "svd")) {
        zend_throw_exception(tsr_ce_exception, "multivariateNormal: only method 'svd' is supported.", 0);
        RETURN_THROWS();
    }
    uint64_t *st = gen_of(Z_OBJ_P(ZEND_THIS))->st;
    const int64_t dim = (int64_t)zend_hash_num_elements(Z_ARRVAL_P(mean_z));
    if (dim < 1) { zend_throw_exception(tsr_ce_exception, "multivariateNormal: mean must be non-empty.", 0); RETURN_THROWS(); }
    if ((int64_t)zend_hash_num_elements(Z_ARRVAL_P(cov_z)) != dim) {
        zend_throw_exception(tsr_ce_exception, "multivariateNormal: cov must be a (d, d) matrix matching mean.", 0); RETURN_THROWS();
    }
    /* validate every row before allocating dim * dim doubles, so a malformed cov fails cheaply */
    { zval *row; ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(cov_z), row) {
        if (Z_TYPE_P(row) != IS_ARRAY || (int64_t)zend_hash_num_elements(Z_ARRVAL_P(row)) != dim) {
            zend_throw_exception(tsr_ce_exception, "multivariateNormal: cov must be a (d, d) matrix matching mean.", 0); RETURN_THROWS();
        }
      } ZEND_HASH_FOREACH_END(); }
    double *mean = (double *)safe_emalloc((size_t)dim, sizeof(double), 0);
    double *cov = (double *)safe_emalloc((size_t)dim, (size_t)dim * sizeof(double), 0);
    { int64_t i = 0; zval *e; ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(mean_z), e) { mean[i++] = zval_get_double(e); } ZEND_HASH_FOREACH_END(); }
    { int64_t r = 0; zval *row;
      ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(cov_z), row) {
        if (Z_TYPE_P(row) != IS_ARRAY || (int64_t)zend_hash_num_elements(Z_ARRVAL_P(row)) != dim) {
            efree(mean); efree(cov); zend_throw_exception(tsr_ce_exception, "multivariateNormal: cov must be a (d, d) matrix matching mean.", 0); RETURN_THROWS();
        }
        int64_t c = 0; zval *e;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(row), e) { cov[r * dim + c] = zval_get_double(e); c++; } ZEND_HASH_FOREACH_END();
        r++;
      } ZEND_HASH_FOREACH_END(); }
    int64_t sdims[32];
    const int snd = size_shape(size, sdims);
    if (snd == -2) { efree(mean); efree(cov); RETURN_THROWS(); }
    int nd; int64_t oshape[40], nrows = 1;
    if (snd < 0) { nd = 1; oshape[0] = dim; nrows = 1; }
    else { nd = snd + 1; for (int d = 0; d < snd; d++) { oshape[d] = sdims[d]; nrows *= sdims[d]; } oshape[snd] = dim; }
    if (tsr_ndarray_new(return_value, TSR_F64, nd, oshape, 0) == FAILURE) { efree(mean); efree(cov); RETURN_THROWS(); }
    tsr_array *o = &Z_TSR_P(return_value)->a;
    double *out = (double *)((char *)o->data + o->offset);
    int rc = tsr_pcg64_multivariate_normal(st, mean, cov, dim, nrows, out);
    efree(mean); efree(cov);
    if (rc < 0) { zval_ptr_dtor(return_value); zend_throw_exception(tsr_ce_exception, "multivariateNormal: SVD did not converge", 0); RETURN_THROWS(); }
}

ZEND_BEGIN_ARG_INFO_EX(ai_default_rng, 0, 0, 0)
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, seed, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_integers, 0, 0, 1)
    ZEND_ARG_INFO(0, low)
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, high, "null")
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, size, "null")
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, endpoint, "false")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_one, 0, 0, 1)
    ZEND_ARG_INFO(0, x)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_choice, 0, 0, 1)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, size, "null")
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, replace, "true")
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, p, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_dirichlet, 0, 0, 1)
    ZEND_ARG_INFO(0, alpha)
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, size, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_multinomial, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, n, IS_LONG, 0)
    ZEND_ARG_INFO(0, pvals)
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, size, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_permuted, 0, 0, 1)
    ZEND_ARG_OBJ_INFO(0, x, Tessero\\Ext\\NDArray, 0)
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, axis, "null")
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, out, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_state, 0, 0, 1)
    ZEND_ARG_INFO(0, words)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_none_g, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(ai_mvhg, 0, 0, 2)
    ZEND_ARG_INFO(0, colors)
    ZEND_ARG_TYPE_INFO(0, nsample, IS_LONG, 0)
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, size, "null")
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, method, "marginals")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(ai_bytes, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, length, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(ai_mvn, 0, 0, 2)
    ZEND_ARG_INFO(0, mean)
    ZEND_ARG_INFO(0, cov)
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, size, "null")
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, checkValid, "'warn'")
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, tol, "1e-8")
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, method, "'svd'")
ZEND_END_ARG_INFO()

static const zend_function_entry generator_methods[] = {
    PHP_ME(Generator, defaultRng, ai_default_rng, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(Generator, integers, ai_integers, ZEND_ACC_PUBLIC)
    PHP_ME(Generator, bytes, ai_bytes, ZEND_ACC_PUBLIC)
    PHP_ME(Generator, multivariateNormal, ai_mvn, ZEND_ACC_PUBLIC)
    PHP_ME(Generator, choice, ai_choice, ZEND_ACC_PUBLIC)
    PHP_ME(Generator, dirichlet, ai_dirichlet, ZEND_ACC_PUBLIC)
    PHP_ME(Generator, multinomial, ai_multinomial, ZEND_ACC_PUBLIC)
    PHP_ME(Generator, permuted, ai_permuted, ZEND_ACC_PUBLIC)
    PHP_ME(Generator, multivariateHypergeometric, ai_mvhg, ZEND_ACC_PUBLIC)
    PHP_ME(Generator, shuffle, ai_one, ZEND_ACC_PUBLIC)
    PHP_ME(Generator, permutation, ai_one, ZEND_ACC_PUBLIC)
    PHP_ME(Generator, getState, ai_none_g, ZEND_ACC_PUBLIC)
    PHP_ME(Generator, setState, ai_state, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

/* called by tsr_register_fn with the registry's Generator methods (kind "random") */
void tsr_register_generator(const zend_function_entry *registry_methods, int n)
{
    const int nfixed = (int)(sizeof generator_methods / sizeof generator_methods[0]) - 1;
    zend_function_entry *all = pecalloc((size_t)(nfixed + n + 1), sizeof(zend_function_entry), 1);
    memcpy(all, generator_methods, sizeof(zend_function_entry) * (size_t)nfixed);
    memcpy(all + nfixed, registry_methods, sizeof(zend_function_entry) * (size_t)n);
    zend_class_entry ce;
    INIT_NS_CLASS_ENTRY(ce, "Tessero\\Ext\\Random", "Generator", all);
    tsr_ce_generator = zend_register_internal_class(&ce);
    tsr_ce_generator->ce_flags |= ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES | ZEND_ACC_NOT_SERIALIZABLE;
    tsr_ce_generator->create_object = gen_create;
    memcpy(&generator_handlers, zend_get_std_object_handlers(), sizeof generator_handlers);
    generator_handlers.offset = XtOffsetOf(tsr_gen, std);
    generator_handlers.clone_obj = gen_clone;
}
