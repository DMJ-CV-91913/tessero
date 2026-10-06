/*
 * Tessero\Ext\Engine: native solvers and generators.
 *
 *   Engine::linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, maxiter)        -> array
 *   Engine::milp(c, integrality, A_ub, b_ub, A_eq, b_eq, bounds, ...)  -> array
 *   Engine::mdpValueIteration(P, R, gamma, epsilon, maxIter)          -> array
 *   Engine::mdpPolicyIteration(P, R, gamma, evalSweeps, maxIter)       -> array
 *   Engine::mdpFiniteHorizon(P, R, horizon, gamma)                     -> array
 *   Engine::fft(x, inverse)                                            -> NDArray (complex128)
 *   Engine::random / normal / integers(seed, shape, ...)               -> NDArray (NumPy's stream)
 *
 * Matrices may be PHP arrays or Tessero\Ext\NDArray. P is (A, S, S) dense, or
 * ['indptr' => ..., 'indices' => ..., 'data' => ...] CSR with A*S rows.
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "zend_exceptions.h"
#include "php_tessero.h"

static const char *LP_STATUS[] = {
    "Optimization terminated successfully.", "Iteration or node limit reached.", "The problem is infeasible.",
    "The problem is unbounded.", "Numerical difficulties encountered.",
};

/* Contiguous float64 (or other dtype) array from a PHP value; `out` owns a reference. */
static int as_contig(zval *v, int dtype, zval *out)
{
    zval tmp;
    if (tsr_ndarray_from_zval(v, dtype, &tmp) == FAILURE) return FAILURE;
    int r = tsr_ndarray_contiguous_as(&tmp, dtype, out);
    zval_ptr_dtor(&tmp);
    return r;
}

static void *ptr0(zval *z) { tsr_obj *o = Z_TSR_P(z); return (char *)o->a.data + o->a.offset; }

static void list_from(zval *rv, const double *p, int64_t n)
{
    array_init_size(rv, (uint32_t)n);
    for (int64_t i = 0; i < n; i++) add_next_index_double(rv, p[i]);
}

typedef struct {
    int64_t n, m_ub, m_eq;
    zval c, Aub, bub, Aeq, beq, lb, ub;
} lp_in;

static void lp_free(lp_in *p)
{
    zval_ptr_dtor(&p->c); zval_ptr_dtor(&p->Aub); zval_ptr_dtor(&p->bub); zval_ptr_dtor(&p->Aeq);
    zval_ptr_dtor(&p->beq); zval_ptr_dtor(&p->lb); zval_ptr_dtor(&p->ub);
}

static int lp_block(zval *A, zval *b, int64_t n, const char *name, zval *Ao, zval *bo, int64_t *m)
{
    *m = 0;
    if ((!A || Z_TYPE_P(A) == IS_NULL) && (!b || Z_TYPE_P(b) == IS_NULL)) return SUCCESS;
    if (!A || !b || Z_TYPE_P(A) == IS_NULL || Z_TYPE_P(b) == IS_NULL) {
        zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s and its right-hand side must be given together", name);
        return FAILURE;
    }
    if (as_contig(A, 0, Ao) == FAILURE || as_contig(b, 0, bo) == FAILURE) return FAILURE;
    tsr_obj *a = Z_TSR_P(Ao);
    int64_t rows = a->a.ndim == 2 ? a->a.shape[0] : (a->a.ndim == 1 && a->a.shape[0] > 0 ? 1 : 0);
    int64_t cols = a->a.ndim == 2 ? a->a.shape[1] : a->a.shape[0];
    if (rows == 0) return SUCCESS;
    if (cols != n || tsr_array_size(&Z_TSR_P(bo)->a) != rows) {
        zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s is %lldx%lld but c has %lld entries / the right-hand side has %lld",
                                name, (long long)rows, (long long)cols, (long long)n, (long long)tsr_array_size(&Z_TSR_P(bo)->a));
        return FAILURE;
    }
    *m = rows;
    return SUCCESS;
}

static int lp_prepare(zval *c, zval *Aub, zval *bub, zval *Aeq, zval *beq, HashTable *bounds, lp_in *p)
{
    ZVAL_UNDEF(&p->c); ZVAL_UNDEF(&p->Aub); ZVAL_UNDEF(&p->bub); ZVAL_UNDEF(&p->Aeq);
    ZVAL_UNDEF(&p->beq); ZVAL_UNDEF(&p->lb); ZVAL_UNDEF(&p->ub);
    if (as_contig(c, 0, &p->c) == FAILURE) return FAILURE;
    p->n = tsr_array_size(&Z_TSR(p->c)->a);
    if (p->n == 0) { zend_throw_exception(tsr_ce_shape_exception, "c must not be empty", 0); return FAILURE; }
    if (lp_block(Aub, bub, p->n, "A_ub", &p->Aub, &p->bub, &p->m_ub) == FAILURE) return FAILURE;
    if (lp_block(Aeq, beq, p->n, "A_eq", &p->Aeq, &p->beq, &p->m_eq) == FAILURE) return FAILURE;
    int64_t shape[1] = {p->n};
    if (tsr_ndarray_new(&p->lb, 0, 1, shape, 0) == FAILURE || tsr_ndarray_new(&p->ub, 0, 1, shape, 0) == FAILURE) return FAILURE;
    double *lo = ptr0(&p->lb), *hi = ptr0(&p->ub);
    for (int64_t j = 0; j < p->n; j++) { lo[j] = 0.0; hi[j] = INFINITY; }
    if (!bounds) return SUCCESS;
    uint32_t nb = zend_hash_num_elements(bounds);
    zval *first = zend_hash_index_find(bounds, 0), *second = zend_hash_index_find(bounds, 1);
    int single = nb == 2 && first && second && Z_TYPE_P(first) != IS_ARRAY && Z_TYPE_P(second) != IS_ARRAY;
    if (!single && (int64_t)nb != p->n) {
        zend_throw_exception_ex(tsr_ce_shape_exception, 0, "bounds must be one [lo, hi] pair or %lld pairs", (long long)p->n);
        return FAILURE;
    }
    int64_t j = 0;
    zval *pair;
    ZEND_HASH_FOREACH_VAL(bounds, pair) {
        zval *l, *h;
        if (single) { l = first; h = second; }
        else {
            ZVAL_DEREF(pair);
            if (Z_TYPE_P(pair) != IS_ARRAY) { zend_throw_exception(tsr_ce_shape_exception, "each bound must be a [lo, hi] pair", 0); return FAILURE; }
            l = zend_hash_index_find(Z_ARRVAL_P(pair), 0);
            h = zend_hash_index_find(Z_ARRVAL_P(pair), 1);
        }
        double lv = !l || Z_TYPE_P(l) == IS_NULL ? -INFINITY : zval_get_double(l);
        double hv = !h || Z_TYPE_P(h) == IS_NULL ? INFINITY : zval_get_double(h);
        if (single) { for (int64_t k = 0; k < p->n; k++) { lo[k] = lv; hi[k] = hv; } break; }
        lo[j] = lv;
        hi[j] = hv;
        j++;
    } ZEND_HASH_FOREACH_END();
    return SUCCESS;
}

#define PD(z) (Z_TYPE(z) == IS_UNDEF ? NULL : (double *)ptr0(&(z)))

PHP_METHOD(Engine, linprog)
{
    zval *c, *Aub = NULL, *bub = NULL, *Aeq = NULL, *beq = NULL;
    HashTable *bounds = NULL;
    zend_long maxiter = 100000;
    ZEND_PARSE_PARAMETERS_START(1, 7)
        Z_PARAM_ZVAL(c)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(Aub)
        Z_PARAM_ZVAL(bub)
        Z_PARAM_ZVAL(Aeq)
        Z_PARAM_ZVAL(beq)
        Z_PARAM_ARRAY_HT_OR_NULL(bounds)
        Z_PARAM_LONG(maxiter)
    ZEND_PARSE_PARAMETERS_END();
    lp_in p;
    if (lp_prepare(c, Aub, bub, Aeq, beq, bounds, &p) == FAILURE) { lp_free(&p); RETURN_THROWS(); }
    double *x = ecalloc((size_t)p.n, sizeof(double)), *red = ecalloc((size_t)p.n, sizeof(double));
    double *yu = ecalloc((size_t)(p.m_ub + 1), sizeof(double)), *ye = ecalloc((size_t)(p.m_eq + 1), sizeof(double));
    double fun = 0;
    int64_t nit = 0;
    int st = tsr_linprog(p.n, PD(p.c), p.m_ub, p.m_ub ? PD(p.Aub) : NULL, p.m_ub ? PD(p.bub) : NULL, p.m_eq, p.m_eq ? PD(p.Aeq) : NULL,
                         p.m_eq ? PD(p.beq) : NULL, PD(p.lb), PD(p.ub), maxiter, 1e-9, x, &fun, yu, ye, red, &nit);
    if (st < 0) {
        if (st == -1) zend_argument_value_error(1, "and the constraints must be finite (no NaN or INF in c, A_ub, b_ub, A_eq, b_eq; no NaN in bounds)"); else tsr_throw_rc(st, "linprog");
    } else {
        array_init(return_value);
        zval zx, zu, ze, zr;
        if (st == 0 || st == 1) { list_from(&zx, x, p.n); add_assoc_zval(return_value, "x", &zx); add_assoc_double(return_value, "fun", fun); }
        else { add_assoc_null(return_value, "x"); add_assoc_null(return_value, "fun"); }
        add_assoc_bool(return_value, "success", st == 0);
        add_assoc_long(return_value, "status", st);
        add_assoc_string(return_value, "message", (char *)LP_STATUS[st <= 4 ? st : 4]);
        add_assoc_long(return_value, "nit", nit);
        list_from(&zu, yu, st == 0 ? p.m_ub : 0);
        list_from(&ze, ye, st == 0 ? p.m_eq : 0);
        list_from(&zr, red, st == 0 ? p.n : 0);
        add_assoc_zval(return_value, "ineqlin", &zu);
        add_assoc_zval(return_value, "eqlin", &ze);
        add_assoc_zval(return_value, "reduced_costs", &zr);
    }
    efree(x); efree(red); efree(yu); efree(ye);
    lp_free(&p);
}

PHP_METHOD(Engine, milp)
{
    zval *c, *integ, *Aub = NULL, *bub = NULL, *Aeq = NULL, *beq = NULL;
    HashTable *bounds = NULL;
    zend_long node_limit = 100000;
    double gap = 0.0;
    ZEND_PARSE_PARAMETERS_START(2, 9)
        Z_PARAM_ZVAL(c)
        Z_PARAM_ZVAL(integ)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(Aub)
        Z_PARAM_ZVAL(bub)
        Z_PARAM_ZVAL(Aeq)
        Z_PARAM_ZVAL(beq)
        Z_PARAM_ARRAY_HT_OR_NULL(bounds)
        Z_PARAM_LONG(node_limit)
        Z_PARAM_DOUBLE(gap)
    ZEND_PARSE_PARAMETERS_END();
    lp_in p;
    if (lp_prepare(c, Aub, bub, Aeq, beq, bounds, &p) == FAILURE) { lp_free(&p); RETURN_THROWS(); }
    uint8_t *flags = ecalloc((size_t)p.n, 1);
    ZVAL_DEREF(integ);
    if (Z_TYPE_P(integ) == IS_TRUE || Z_TYPE_P(integ) == IS_FALSE) {
        memset(flags, Z_TYPE_P(integ) == IS_TRUE, (size_t)p.n);
    } else if (Z_TYPE_P(integ) == IS_ARRAY && (int64_t)zend_hash_num_elements(Z_ARRVAL_P(integ)) == p.n) {
        int64_t j = 0;
        zval *e;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(integ), e) { flags[j++] = zend_is_true(e); } ZEND_HASH_FOREACH_END();
    } else {
        efree(flags);
        lp_free(&p);
        zend_throw_exception(tsr_ce_shape_exception, "integrality must be a bool or one flag per variable", 0);
        RETURN_THROWS();
    }
    double *x = ecalloc((size_t)p.n, sizeof(double));
    double fun = INFINITY, bound = -INFINITY;
    int64_t nodes = 0;
    int st = tsr_milp(p.n, PD(p.c), p.m_ub, p.m_ub ? PD(p.Aub) : NULL, p.m_ub ? PD(p.bub) : NULL, p.m_eq, p.m_eq ? PD(p.Aeq) : NULL,
                      p.m_eq ? PD(p.beq) : NULL, PD(p.lb), PD(p.ub), flags, node_limit, gap, 1e-9, x, &fun, &nodes, &bound);
    if (st < 0) {
        if (st == -1) zend_argument_value_error(1, "and the constraints must be finite (no NaN or INF in c, A_ub, b_ub, A_eq, b_eq; no NaN in bounds)"); else tsr_throw_rc(st, "milp");
    } else {
        array_init(return_value);
        int has = st == 0 || (st == 1 && fun < INFINITY);
        if (has) { zval zx; list_from(&zx, x, p.n); add_assoc_zval(return_value, "x", &zx); add_assoc_double(return_value, "fun", fun); }
        else { add_assoc_null(return_value, "x"); add_assoc_null(return_value, "fun"); }
        add_assoc_bool(return_value, "success", st == 0);
        add_assoc_long(return_value, "status", st);
        add_assoc_string(return_value, "message", (char *)LP_STATUS[st <= 4 ? st : 4]);
        add_assoc_long(return_value, "nodes", nodes);
        add_assoc_double(return_value, "best_bound", bound);
    }
    efree(x);
    efree(flags);
    lp_free(&p);
}

/* ================================================================ MDP */

typedef struct {
    int64_t S, A;
    zval indptr, indices, data, R;
} mdp_in;

static void mdp_free(mdp_in *m)
{
    zval_ptr_dtor(&m->indptr); zval_ptr_dtor(&m->indices); zval_ptr_dtor(&m->data); zval_ptr_dtor(&m->R);
}

/* R finite; every row of P a probability distribution (finite, >= 0, sums to 1 within 1e-8). */
static int mdp_validate(mdp_in *m)
{
    const double *rp = ptr0(&m->R);
    for (int64_t k = 0; k < m->S * m->A; k++) {
        if (!isfinite(rp[k])) { zend_throw_exception(tsr_ce_exception, "Rewards must be finite (no NaN or INF)", 0); return FAILURE; }
    }
    const int64_t rows = m->A * m->S;
    const int64_t *ipp = ptr0(&m->indptr);
    const double *dp = ptr0(&m->data);
    for (int64_t row = 0; row < rows; row++) {
        double s = 0;
        for (int64_t k = ipp[row]; k < ipp[row + 1]; k++) {
            if (!(dp[k] >= 0) || !isfinite(dp[k])) {
                zend_throw_exception(tsr_ce_exception, "Transition probabilities must be finite and non-negative", 0);
                return FAILURE;
            }
            s += dp[k];
        }
        if (fabs(s - 1.0) > 1e-8) {
            zend_throw_exception_ex(tsr_ce_exception, 0, "Transition probabilities of state %lld under action %lld sum to %.12g, not 1",
                                    (long long)(row % m->S), (long long)(row / m->S), s);
            return FAILURE;
        }
    }
    return SUCCESS;
}

static int mdp_prepare(zval *P, zval *R, mdp_in *m)
{
    ZVAL_UNDEF(&m->indptr); ZVAL_UNDEF(&m->indices); ZVAL_UNDEF(&m->data); ZVAL_UNDEF(&m->R);
    if (as_contig(R, 0, &m->R) == FAILURE) return FAILURE;
    tsr_obj *r = Z_TSR(m->R);
    if (r->a.ndim != 2) { zend_throw_exception(tsr_ce_shape_exception, "R must have shape (states, actions)", 0); return FAILURE; }
    m->S = r->a.shape[0];
    m->A = r->a.shape[1];
    ZVAL_DEREF(P);
    zval *ip = Z_TYPE_P(P) == IS_ARRAY ? zend_hash_str_find(Z_ARRVAL_P(P), "indptr", 6) : NULL;
    if (ip) {
        zval *ix = zend_hash_str_find(Z_ARRVAL_P(P), "indices", 7), *dx = zend_hash_str_find(Z_ARRVAL_P(P), "data", 4);
        if (!ix || !dx) { zend_throw_exception(tsr_ce_shape_exception, "CSR transitions need indptr, indices and data", 0); return FAILURE; }
        if (as_contig(ip, 2, &m->indptr) == FAILURE || as_contig(ix, 2, &m->indices) == FAILURE || as_contig(dx, 0, &m->data) == FAILURE) return FAILURE;
        if (tsr_array_size(&Z_TSR(m->indptr)->a) != m->A * m->S + 1) {
            zend_throw_exception(tsr_ce_shape_exception, "indptr must have actions * states + 1 entries", 0);
            return FAILURE;
        }
        const int64_t *ipc = ptr0(&m->indptr);
        int64_t nnz = tsr_array_size(&Z_TSR(m->data)->a);
        if (ipc[0] != 0 || ipc[m->A * m->S] != nnz || tsr_array_size(&Z_TSR(m->indices)->a) != nnz) {
            zend_throw_exception(tsr_ce_shape_exception, "CSR transitions: indptr must start at 0 and end at len(data) == len(indices)", 0);
            return FAILURE;
        }
        for (int64_t row = 0; row < m->A * m->S; row++) {
            if (ipc[row + 1] < ipc[row]) { zend_throw_exception(tsr_ce_shape_exception, "CSR transitions: indptr must be non-decreasing", 0); return FAILURE; }
        }
        return mdp_validate(m);
    }
    zval D;
    if (as_contig(P, 0, &D) == FAILURE) return FAILURE;
    tsr_obj *d = Z_TSR(D);
    if (d->a.ndim != 3 || d->a.shape[0] != m->A || d->a.shape[1] != m->S || d->a.shape[2] != m->S) {
        zval_ptr_dtor(&D);
        zend_throw_exception_ex(tsr_ce_shape_exception, 0, "P must have shape (%lld, %lld, %lld) to match R", (long long)m->A, (long long)m->S, (long long)m->S);
        return FAILURE;
    }
    int64_t rows = m->A * m->S;
    const double *pd = ptr0(&D);
    int64_t nnz = tsr_dense_nnz(rows, m->S, pd);
    int64_t s1[1] = {rows + 1}, s2[1] = {nnz};
    if (tsr_ndarray_new(&m->indptr, 2, 1, s1, 0) == FAILURE || tsr_ndarray_new(&m->indices, 2, 1, s2, 0) == FAILURE ||
        tsr_ndarray_new(&m->data, 0, 1, s2, 0) == FAILURE) { zval_ptr_dtor(&D); return FAILURE; }
    tsr_dense_to_csr(rows, m->S, pd, ptr0(&m->indptr), ptr0(&m->indices), ptr0(&m->data));
    zval_ptr_dtor(&D);
    return mdp_validate(m);
}

#define MDP_ARGS(m) (m).S, (m).A, ptr0(&(m).indptr), ptr0(&(m).indices), ptr0(&(m).data), ptr0(&(m).R)

PHP_METHOD(Engine, mdpValueIteration)
{
    zval *P, *R;
    double gamma, eps = 0;
    zend_long maxiter = 10000;
    ZEND_PARSE_PARAMETERS_START(3, 5)
        Z_PARAM_ZVAL(P)
        Z_PARAM_ZVAL(R)
        Z_PARAM_DOUBLE(gamma)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(eps)
        Z_PARAM_LONG(maxiter)
    ZEND_PARSE_PARAMETERS_END();
    if (gamma < 0 || gamma > 1) { zend_argument_value_error(3, "must be in [0, 1]"); RETURN_THROWS(); }
    if (eps <= 0) eps = TESSERO_G(epsilon);
    mdp_in m;
    if (mdp_prepare(P, R, &m) == FAILURE) { mdp_free(&m); RETURN_THROWS(); }
    zval V, pol;
    int64_t s[1] = {m.S};
    tsr_ndarray_new(&V, 0, 1, s, 1);
    tsr_ndarray_new(&pol, 2, 1, s, 1);
    int64_t it = 0;
    double delta = 0;
    int st = tsr_mdp_value_iteration(MDP_ARGS(m), gamma, eps, maxiter, ptr0(&V), ptr0(&pol), &it, &delta);
    mdp_free(&m);
    if (st < 0) { zval_ptr_dtor(&V); zval_ptr_dtor(&pol); tsr_throw_rc(st, "value iteration"); RETURN_THROWS(); }
    array_init(return_value);
    add_assoc_zval(return_value, "values", &V);
    add_assoc_zval(return_value, "policy", &pol);
    add_assoc_long(return_value, "iterations", it);
    add_assoc_bool(return_value, "converged", st == 0);
    add_assoc_double(return_value, "delta", delta);
}

PHP_METHOD(Engine, mdpPolicyIteration)
{
    zval *P, *R;
    double gamma;
    zend_long sweeps = 0, maxiter = 1000;
    ZEND_PARSE_PARAMETERS_START(3, 5)
        Z_PARAM_ZVAL(P)
        Z_PARAM_ZVAL(R)
        Z_PARAM_DOUBLE(gamma)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(sweeps)
        Z_PARAM_LONG(maxiter)
    ZEND_PARSE_PARAMETERS_END();
    if (gamma < 0 || gamma > 1 || (sweeps == 0 && gamma >= 1)) { zend_argument_value_error(3, "must be in [0, 1) for exact policy iteration"); RETURN_THROWS(); }
    mdp_in m;
    if (mdp_prepare(P, R, &m) == FAILURE) { mdp_free(&m); RETURN_THROWS(); }
    zval V, pol;
    int64_t s[1] = {m.S};
    tsr_ndarray_new(&V, 0, 1, s, 1);
    tsr_ndarray_new(&pol, 2, 1, s, 1);
    int64_t it = 0;
    int st = tsr_mdp_policy_iteration(MDP_ARGS(m), gamma, sweeps, TESSERO_G(epsilon), maxiter, ptr0(&V), ptr0(&pol), &it);
    mdp_free(&m);
    if (st < 0) { zval_ptr_dtor(&V); zval_ptr_dtor(&pol); tsr_throw_rc(st, "policy iteration"); RETURN_THROWS(); }
    array_init(return_value);
    add_assoc_zval(return_value, "values", &V);
    add_assoc_zval(return_value, "policy", &pol);
    add_assoc_long(return_value, "iterations", it);
    add_assoc_bool(return_value, "converged", st == 0);
}

PHP_METHOD(Engine, mdpFiniteHorizon)
{
    zval *P, *R;
    zend_long T;
    double gamma = 1.0;
    ZEND_PARSE_PARAMETERS_START(3, 4)
        Z_PARAM_ZVAL(P)
        Z_PARAM_ZVAL(R)
        Z_PARAM_LONG(T)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(gamma)
    ZEND_PARSE_PARAMETERS_END();
    if (T < 1) { zend_argument_value_error(3, "must be at least 1"); RETURN_THROWS(); }
    mdp_in m;
    if (mdp_prepare(P, R, &m) == FAILURE) { mdp_free(&m); RETURN_THROWS(); }
    zval V, pol;
    int64_t sv[2] = {T + 1, m.S}, sp[2] = {T, m.S};
    tsr_ndarray_new(&V, 0, 2, sv, 1);
    tsr_ndarray_new(&pol, 2, 2, sp, 1);
    int st = tsr_mdp_finite_horizon(MDP_ARGS(m), gamma, T, NULL, ptr0(&V), ptr0(&pol));
    mdp_free(&m);
    if (st < 0) { zval_ptr_dtor(&V); zval_ptr_dtor(&pol); tsr_throw_rc(st, "finite horizon"); RETURN_THROWS(); }
    array_init(return_value);
    add_assoc_zval(return_value, "values", &V);
    add_assoc_zval(return_value, "policy", &pol);
}

/* ================================================================ FFT and random numbers */

PHP_METHOD(Engine, fft)
{
    zval *x;
    zend_bool inverse = 0;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(x)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(inverse)
    ZEND_PARSE_PARAMETERS_END();
    if (as_contig(x, 6, return_value) == FAILURE) RETURN_THROWS();
    /* as_contig may have returned the caller's own contiguous complex array: transform a copy */
    tsr_obj *o = Z_TSR_P(return_value);
    if (o->a.ndim == 0) { zend_throw_exception(tsr_ce_shape_exception, "fft needs at least one dimension", 0); RETURN_THROWS(); }
    zval C;
    int64_t n = o->a.shape[o->a.ndim - 1];
    tsr_ndarray_new(&C, 6, o->a.ndim, o->a.shape, 0);
    memcpy(ptr0(&C), ptr0(return_value), (size_t)(tsr_array_size(&o->a) * 16));
    zval_ptr_dtor(return_value);
    ZVAL_COPY_VALUE(return_value, &C);
    int64_t rows = n ? tsr_array_size(&Z_TSR_P(return_value)->a) / n : 0;
    if (rows > 0) {
        double *p = ptr0(return_value);
        int rc = tsr_fft(n, rows, inverse, p, p);
        if (rc != 0) { tsr_throw_rc(rc, "fft"); RETURN_THROWS(); }
    }
}

static int rng_state(zend_long seed, uint64_t st[6])
{
    if (seed < 0) { zend_argument_value_error(1, "must be a non-negative integer"); return FAILURE; }
    uint32_t words[2];
    int n = 1;
    words[0] = (uint32_t)(seed & 0xffffffff);
    if ((uint64_t)seed > 0xffffffffULL) { words[1] = (uint32_t)((uint64_t)seed >> 32); n = 2; }
    uint64_t pool[4];
    tsr_seed_sequence(words, n, pool, 4);
    tsr_pcg64_seed(st, pool[0], pool[1], pool[2], pool[3]);
    return SUCCESS;
}

static int shape_arg(zval *z, int64_t *shape, int32_t *nd)
{
    ZVAL_DEREF(z);
    if (Z_TYPE_P(z) == IS_LONG) { shape[0] = Z_LVAL_P(z); *nd = 1; return shape[0] >= 0 ? SUCCESS : FAILURE; }
    if (Z_TYPE_P(z) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_P(z)) > 32) return FAILURE;
    int32_t k = 0;
    zval *e;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(z), e) { if (Z_TYPE_P(e) != IS_LONG || Z_LVAL_P(e) < 0) return FAILURE; shape[k++] = Z_LVAL_P(e); } ZEND_HASH_FOREACH_END();
    *nd = k;
    return SUCCESS;
}

static void rng_method(INTERNAL_FUNCTION_PARAMETERS, int kind)
{
    zend_long seed;
    zval *shp;
    double a = 0.0, b = 1.0;
    zend_long lo = 0, hi = 0;
    if (kind == 2) {
        ZEND_PARSE_PARAMETERS_START(4, 4)
            Z_PARAM_LONG(seed)
            Z_PARAM_ZVAL(shp)
            Z_PARAM_LONG(lo)
            Z_PARAM_LONG(hi)
        ZEND_PARSE_PARAMETERS_END();
    } else {
        ZEND_PARSE_PARAMETERS_START(2, 4)
            Z_PARAM_LONG(seed)
            Z_PARAM_ZVAL(shp)
            Z_PARAM_OPTIONAL
            Z_PARAM_DOUBLE(a)
            Z_PARAM_DOUBLE(b)
        ZEND_PARSE_PARAMETERS_END();
    }
    int64_t shape[32];
    int32_t nd;
    if (shape_arg(shp, shape, &nd) == FAILURE) { zend_argument_value_error(2, "must be a size or a shape"); RETURN_THROWS(); }
    if (kind == 2 && hi <= lo) { zend_argument_value_error(4, "must be greater than low"); RETURN_THROWS(); }
    uint64_t st[6];
    if (rng_state(seed, st) == FAILURE) RETURN_THROWS();
    if (tsr_ndarray_new(return_value, kind == 2 ? 2 : 0, nd, shape, 0) == FAILURE) RETURN_THROWS();
    int64_t n = tsr_array_size(&Z_TSR_P(return_value)->a);
    if (n == 0) return;
    if (kind == 0) {
        double *p = ptr0(return_value);
        tsr_pcg64_random(st, n, p);
        if (a != 0.0 || b != 1.0) for (int64_t i = 0; i < n; i++) p[i] = a + (b - a) * p[i];
    } else if (kind == 1) {
        tsr_pcg64_normal(st, n, a, b, ptr0(return_value));
    } else {
        tsr_pcg64_integers(st, n, lo, hi, ptr0(return_value));
    }
}

PHP_METHOD(Engine, random) { rng_method(INTERNAL_FUNCTION_PARAM_PASSTHRU, 0); }
PHP_METHOD(Engine, normal) { rng_method(INTERNAL_FUNCTION_PARAM_PASSTHRU, 1); }
PHP_METHOD(Engine, integers) { rng_method(INTERNAL_FUNCTION_PARAM_PASSTHRU, 2); }

/* method table entries live in tessero.c (engine_methods) */
