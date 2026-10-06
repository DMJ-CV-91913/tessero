/*
 * tessero extension: module lifecycle, INI settings, exceptions,
 * Tessero\Ext\Engine (settings) and Tessero\Ext\Operand (operators for
 * userland classes).
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "php_ini.h"
#include "ext/standard/info.h"
#include "zend_exceptions.h"
#include "ext/spl/spl_exceptions.h"
#include "php_tessero.h"
#include "tessero_arginfo.h"

ZEND_DECLARE_MODULE_GLOBALS(tessero)

zend_class_entry *tsr_ce_engine, *tsr_ce_operand;
zend_class_entry *tsr_ce_exception, *tsr_ce_shape_exception, *tsr_ce_index_exception, *tsr_ce_dtype_exception,
    *tsr_ce_memory_exception;

/* ================================================================ INI */

static ZEND_INI_MH(OnUpdateThreads)
{
    zend_long n = ZEND_ATOL(ZSTR_VAL(new_value));
    if (n < 1) n = 1;
    TESSERO_G(threads) = n;
    tsr_set_threads((int)n);
    return SUCCESS;
}

static ZEND_INI_MH(OnUpdateBudget)
{
    zend_long n = zend_ini_parse_quantity_warn(new_value, entry->name);
    if (n < 0) n = 0;
    TESSERO_G(memory_budget) = n;
    tsr_set_budget(n);
    return SUCCESS;
}

PHP_INI_BEGIN()
    PHP_INI_ENTRY("tessero.threads", "1", PHP_INI_ALL, OnUpdateThreads)
    PHP_INI_ENTRY("tessero.memory_budget", "0", PHP_INI_ALL, OnUpdateBudget)
    STD_PHP_INI_ENTRY("tessero.epsilon", "1e-6", PHP_INI_ALL, OnUpdateReal, epsilon, zend_tessero_globals, tessero_globals)
PHP_INI_END()

static PHP_GINIT_FUNCTION(tessero)
{
#if defined(COMPILE_DL_TESSERO) && defined(ZTS)
    ZEND_TSRMLS_CACHE_UPDATE();
#endif
    tessero_globals->threads = 1;
    tessero_globals->memory_budget = 0;
    tessero_globals->epsilon = 1e-6;
}

/* ================================================================ Engine (settings) */

PHP_METHOD(Engine, version) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_STRING(PHP_TESSERO_VERSION); }
PHP_METHOD(Engine, kernelVersion) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_STRING(tsr_version()); }

PHP_METHOD(Engine, simd)
{
    ZEND_PARSE_PARAMETERS_NONE();
    static const char *names[] = {"baseline", "avx2", "avx512", "neon"};
    int l = tsr_simd_level();
    RETURN_STRING(l >= 0 && l < 4 ? names[l] : "unknown");
}

PHP_METHOD(Engine, openmp) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_BOOL(tsr_openmp()); }

PHP_METHOD(Engine, setMaxThreads)
{
    zend_long n;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(n)
    ZEND_PARSE_PARAMETERS_END();
    if (n < 1) { zend_argument_value_error(1, "must be at least 1"); RETURN_THROWS(); }
    TESSERO_G(threads) = n;
    tsr_set_threads((int)n);
}

PHP_METHOD(Engine, getMaxThreads) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_LONG(tsr_get_threads()); }

PHP_METHOD(Engine, setEpsilon)
{
    double e;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_DOUBLE(e)
    ZEND_PARSE_PARAMETERS_END();
    if (!(e > 0)) { zend_argument_value_error(1, "must be positive"); RETURN_THROWS(); }
    TESSERO_G(epsilon) = e;
}

PHP_METHOD(Engine, getEpsilon) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_DOUBLE(TESSERO_G(epsilon)); }

PHP_METHOD(Engine, setMemoryBudget)
{
    zend_long b;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(b)
    ZEND_PARSE_PARAMETERS_END();
    if (b < 0) b = 0;
    TESSERO_G(memory_budget) = b;
    tsr_set_budget(b);
}

PHP_METHOD(Engine, memoryBudget) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_LONG(tsr_budget()); }
PHP_METHOD(Engine, memoryInUse) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_LONG(tsr_allocated()); }
PHP_METHOD(Engine, peakMemory) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_LONG(tsr_peak()); }

PHP_METHOD(Engine, info)
{
    ZEND_PARSE_PARAMETERS_NONE();
    static const char *names[] = {"baseline", "avx2", "avx512", "neon"};
    array_init(return_value);
    add_assoc_string(return_value, "extension", PHP_TESSERO_VERSION);
    add_assoc_string(return_value, "kernel", (char *)tsr_version());
    add_assoc_string(return_value, "simd", (char *)names[tsr_simd_level() & 3]);
    add_assoc_bool(return_value, "openmp", tsr_openmp());
    add_assoc_long(return_value, "threads", tsr_get_threads());
    add_assoc_long(return_value, "memory_in_use", tsr_allocated());
    add_assoc_long(return_value, "memory_peak", tsr_peak());
    add_assoc_long(return_value, "memory_budget", tsr_budget());
    add_assoc_long(return_value, "memory_mapped", tsr_memmap_bytes());   /* file-backed, outside the budget */
#ifdef ZTS
    add_assoc_bool(return_value, "zts", 1);
#else
    add_assoc_bool(return_value, "zts", 0);
#endif
}

ZEND_BEGIN_ARG_INFO_EX(ai_e_none, 0, 0, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_e_long, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, value, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_e_double, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, value, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()

#define E_ME(name, ai) PHP_ME(Engine, name, ai, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
static const zend_function_entry engine_methods[] = {
    E_ME(version, ai_e_none)
    E_ME(kernelVersion, ai_e_none)
    E_ME(simd, ai_e_none)
    E_ME(openmp, ai_e_none)
    E_ME(setMaxThreads, ai_e_long)
    E_ME(getMaxThreads, ai_e_none)
    E_ME(setEpsilon, ai_e_double)
    E_ME(getEpsilon, ai_e_none)
    E_ME(setMemoryBudget, ai_e_long)
    E_ME(memoryBudget, ai_e_none)
    E_ME(memoryInUse, ai_e_none)
    E_ME(peakMemory, ai_e_none)
    E_ME(info, ai_e_none)
    E_ME(linprog, ai_linprog)
    E_ME(milp, ai_milp)
    E_ME(mdpValueIteration, ai_vi)
    E_ME(mdpPolicyIteration, ai_pi)
    E_ME(mdpFiniteHorizon, ai_fh)
    E_ME(fft, ai_fft)
    E_ME(random, ai_rand)
    E_ME(normal, ai_rand)
    E_ME(integers, ai_ints)
    PHP_FE_END
};

/* ================================================================ Operand (operators for userland classes) */

static zend_object_handlers operand_handlers;

static zend_object *operand_create(zend_class_entry *ce)
{
    zend_object *obj = zend_objects_new(ce);
    object_properties_init(obj, ce);
    obj->handlers = &operand_handlers;
    return obj;
}

static int is_operand(zval *zv)
{
    return Z_TYPE_P(zv) == IS_OBJECT && instanceof_function(Z_OBJCE_P(zv), tsr_ce_operand);
}

static int call_method(zval *obj, const char *name, zval *arg, zval *ret)
{
    zend_string *fname = zend_string_init(name, strlen(name), 0);
    zend_function *fn = zend_hash_find_ptr_lc(&Z_OBJCE_P(obj)->function_table, fname);
    zend_string_release(fname);
    if (fn == NULL) return FAILURE;
    zend_call_known_instance_method_with_1_params(fn, Z_OBJ_P(obj), ret, arg);
    return SUCCESS;
}

static zend_result operand_do_operation(uint8_t opcode, zval *result, zval *op1, zval *op2)
{
    const char *direct = NULL, *reflected = NULL;
    switch (opcode) {
    case ZEND_ADD: direct = "add"; reflected = "add"; break;
    case ZEND_MUL: direct = "mul"; reflected = "mul"; break;
    case ZEND_SUB: direct = "sub"; reflected = "rsub"; break;
    case ZEND_DIV: direct = "div"; reflected = "rdiv"; break;
    case ZEND_POW: direct = "pow"; reflected = "rpow"; break;
    case ZEND_MOD: direct = "mod"; reflected = NULL; break;
    default: return FAILURE;
    }
    zval tmp;
    ZVAL_UNDEF(&tmp);
    int rc;
    if (is_operand(op1)) rc = call_method(op1, direct, op2, &tmp);
    else if (is_operand(op2) && reflected) rc = call_method(op2, reflected, op1, &tmp);
    else return FAILURE;
    if (rc == FAILURE) return FAILURE;
    if (result == op1) zval_ptr_dtor(op1);
    if (Z_TYPE(tmp) == IS_UNDEF) ZVAL_NULL(result);
    else ZVAL_COPY_VALUE(result, &tmp);
    return SUCCESS;
}

/* ================================================================ module */

PHP_MINIT_FUNCTION(tessero)
{
    REGISTER_INI_ENTRIES();
    zend_class_entry ce;

    INIT_NS_CLASS_ENTRY(ce, "Tessero\\Ext", "Exception", NULL);
    tsr_ce_exception = zend_register_internal_class_ex(&ce, spl_ce_RuntimeException);
    INIT_NS_CLASS_ENTRY(ce, "Tessero\\Ext", "ShapeException", NULL);
    tsr_ce_shape_exception = zend_register_internal_class_ex(&ce, tsr_ce_exception);
    INIT_NS_CLASS_ENTRY(ce, "Tessero\\Ext", "IndexException", NULL);
    tsr_ce_index_exception = zend_register_internal_class_ex(&ce, tsr_ce_exception);
    INIT_NS_CLASS_ENTRY(ce, "Tessero\\Ext", "DTypeException", NULL);
    tsr_ce_dtype_exception = zend_register_internal_class_ex(&ce, tsr_ce_exception);
    INIT_NS_CLASS_ENTRY(ce, "Tessero\\Ext", "MemoryException", NULL);
    tsr_ce_memory_exception = zend_register_internal_class_ex(&ce, tsr_ce_exception);

    INIT_NS_CLASS_ENTRY(ce, "Tessero\\Ext", "Operand", NULL);
    tsr_ce_operand = zend_register_internal_class(&ce);
    tsr_ce_operand->ce_flags |= ZEND_ACC_EXPLICIT_ABSTRACT_CLASS;
    tsr_ce_operand->create_object = operand_create;
    memcpy(&operand_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    operand_handlers.do_operation = operand_do_operation;

    INIT_NS_CLASS_ENTRY(ce, "Tessero\\Ext", "Engine", engine_methods);
    tsr_ce_engine = zend_register_internal_class(&ce);
    tsr_ce_engine->ce_flags |= ZEND_ACC_FINAL;

    tsr_register_ndarray();
    tsr_register_math();
    tsr_register_fn();
    tsr_register_dist();
    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(tessero)
{
    UNREGISTER_INI_ENTRIES();
    return SUCCESS;
}

PHP_RINIT_FUNCTION(tessero)
{
#if defined(COMPILE_DL_TESSERO) && defined(ZTS)
    ZEND_TSRMLS_CACHE_UPDATE();
#endif
    /* each request starts from its configured settings (Octane/FrankenPHP workers included) */
    tsr_set_threads((int)TESSERO_G(threads));
    tsr_set_budget(TESSERO_G(memory_budget));
    return SUCCESS;
}

PHP_MINFO_FUNCTION(tessero)
{
    static const char *names[] = {"baseline", "AVX2", "AVX-512", "NEON"};
    char buf[32];
    php_info_print_table_start();
    php_info_print_table_row(2, "Tessero native extension", "enabled");
    php_info_print_table_row(2, "Version", PHP_TESSERO_VERSION);
    php_info_print_table_row(2, "libtessero", tsr_version());
    php_info_print_table_row(2, "SIMD dispatch", names[tsr_simd_level() & 3]);
    php_info_print_table_row(2, "OpenMP", tsr_openmp() ? "yes" : "no");
    snprintf(buf, sizeof(buf), "%lld", (long long)tsr_allocated());
    php_info_print_table_row(2, "Native memory in use (bytes)", buf);
    php_info_print_table_end();
    DISPLAY_INI_ENTRIES();
}

static const zend_module_dep tessero_deps[] = {
    ZEND_MOD_REQUIRED("json")
    ZEND_MOD_REQUIRED("spl")
    ZEND_MOD_END
};

zend_module_entry tessero_module_entry = {
    STANDARD_MODULE_HEADER_EX,
    NULL,
    tessero_deps,
    "tessero",
    NULL,
    PHP_MINIT(tessero),
    PHP_MSHUTDOWN(tessero),
    PHP_RINIT(tessero),
    NULL,
    PHP_MINFO(tessero),
    PHP_TESSERO_VERSION,
    PHP_MODULE_GLOBALS(tessero),
    PHP_GINIT(tessero),
    NULL,
    NULL,
    STANDARD_MODULE_PROPERTIES_EX
};

#ifdef COMPILE_DL_TESSERO
#ifdef ZTS
ZEND_TSRMLS_CACHE_DEFINE()
#endif
ZEND_GET_MODULE(tessero)
#endif
