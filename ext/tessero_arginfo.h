/* arginfo for Tessero\\Ext\\Engine solver methods (shared by tessero.c) */
#ifndef TESSERO_ARGINFO_H
#define TESSERO_ARGINFO_H
ZEND_BEGIN_ARG_INFO_EX(ai_linprog, 0, 0, 1)
    ZEND_ARG_INFO(0, c)
    ZEND_ARG_INFO(0, A_ub)
    ZEND_ARG_INFO(0, b_ub)
    ZEND_ARG_INFO(0, A_eq)
    ZEND_ARG_INFO(0, b_eq)
    ZEND_ARG_TYPE_INFO(0, bounds, IS_ARRAY, 1)
    ZEND_ARG_TYPE_INFO(0, maxiter, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_milp, 0, 0, 2)
    ZEND_ARG_INFO(0, c)
    ZEND_ARG_INFO(0, integrality)
    ZEND_ARG_INFO(0, A_ub)
    ZEND_ARG_INFO(0, b_ub)
    ZEND_ARG_INFO(0, A_eq)
    ZEND_ARG_INFO(0, b_eq)
    ZEND_ARG_TYPE_INFO(0, bounds, IS_ARRAY, 1)
    ZEND_ARG_TYPE_INFO(0, nodeLimit, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, mipRelGap, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_vi, 0, 0, 3)
    ZEND_ARG_INFO(0, P)
    ZEND_ARG_INFO(0, R)
    ZEND_ARG_TYPE_INFO(0, gamma, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, epsilon, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, maxIter, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_pi, 0, 0, 3)
    ZEND_ARG_INFO(0, P)
    ZEND_ARG_INFO(0, R)
    ZEND_ARG_TYPE_INFO(0, gamma, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, evalSweeps, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, maxIter, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_fh, 0, 0, 3)
    ZEND_ARG_INFO(0, P)
    ZEND_ARG_INFO(0, R)
    ZEND_ARG_TYPE_INFO(0, horizon, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, gamma, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_fft, 0, 0, 1)
    ZEND_ARG_INFO(0, x)
    ZEND_ARG_TYPE_INFO(0, inverse, _IS_BOOL, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_rand, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, seed, IS_LONG, 0)
    ZEND_ARG_INFO(0, shape)
    ZEND_ARG_TYPE_INFO(0, a, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, b, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_ints, 0, 0, 4)
    ZEND_ARG_TYPE_INFO(0, seed, IS_LONG, 0)
    ZEND_ARG_INFO(0, shape)
    ZEND_ARG_TYPE_INFO(0, low, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, high, IS_LONG, 0)
ZEND_END_ARG_INFO()


PHP_METHOD(Engine, linprog);
PHP_METHOD(Engine, milp);
PHP_METHOD(Engine, mdpValueIteration);
PHP_METHOD(Engine, mdpPolicyIteration);
PHP_METHOD(Engine, mdpFiniteHorizon);
PHP_METHOD(Engine, fft);
PHP_METHOD(Engine, random);
PHP_METHOD(Engine, normal);
PHP_METHOD(Engine, integers);
#endif
