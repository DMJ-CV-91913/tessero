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
ZEND_BEGIN_ARG_INFO_EX(ai_state, 0, 0, 1)
    ZEND_ARG_INFO(0, words)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_none_g, 0, 0, 0)
ZEND_END_ARG_INFO()

static const zend_function_entry generator_methods[] = {
    PHP_ME(Generator, defaultRng, ai_default_rng, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(Generator, integers, ai_integers, ZEND_ACC_PUBLIC)
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
