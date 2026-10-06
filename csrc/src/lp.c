/*
 * Linear and mixed-integer programming in C.
 *
 *   minimise  c.x   subject to  A_ub x <= b_ub,  A_eq x = b_eq,  lb <= x <= ub
 *
 * tsr_linprog: dense two-phase primal simplex on a full tableau.
 *   - bounds are removed by substitution (shift to lb, mirror from ub, split
 *     free variables), finite upper bounds of shifted variables become rows;
 *   - Dantzig pricing, switching to Bland's rule after a run of degenerate
 *     pivots (no cycling), Harris two-pass ratio test (stable pivots);
 *   - returns x, the objective, row duals (d objective / d b, SciPy's
 *     "marginals": <= 0 for A_ub rows of a minimisation) and reduced costs.
 *
 * tsr_milp: depth-first branch and bound on tsr_linprog with most-fractional
 *   branching, nearer child first, bound pruning against the incumbent.
 *
 * Status codes: 0 optimal, 1 iteration/node limit, 2 infeasible, 3 unbounded,
 * 4 numerical trouble; negative values are argument/memory errors.
 */
#include "internal.h"
#include <math.h>
#include <stdlib.h>

enum { LP_OPTIMAL = 0, LP_LIMIT = 1, LP_INFEASIBLE = 2, LP_UNBOUNDED = 3, LP_NUMERICAL = 4 };
enum { V_SHIFT = 0, V_MIRROR = 1, V_FREE = 2 };

typedef struct {
    int64_t m, w;   /* constraint rows, columns (excluding rhs) */
    double *T;      /* (m + 1) x (w + 1); row m = reduced costs, T[m][w] = -objective */
    int64_t *basis; /* basic column of each row */
    uint8_t *banned;/* columns that may not enter */
    int64_t nit, maxiter;
    double tol;
} tab_t;

#define AT(t, i, j) ((t)->T[(i) * ((t)->w + 1) + (j)])

static void pivot(tab_t *t, int64_t r, int64_t q)
{
    const int64_t W = t->w + 1;
    double *pr = t->T + r * W;
    const double inv = 1.0 / pr[q];
    for (int64_t j = 0; j < W; j++) pr[j] *= inv;
    pr[q] = 1.0;
    for (int64_t i = 0; i <= t->m; i++) {
        if (i == r) continue;
        double *pi = t->T + i * W;
        const double f = pi[q];
        if (f == 0.0) continue;
        for (int64_t j = 0; j < W; j++) pi[j] -= f * pr[j];
        pi[q] = 0.0;
    }
    t->basis[r] = q;
    t->nit++;
}

/* Run the simplex on the current objective row. Returns LP_OPTIMAL, LP_UNBOUNDED or LP_LIMIT. */
static int simplex(tab_t *t)
{
    const int64_t m = t->m, w = t->w;
    int64_t degenerate = 0;
    for (;;) {
        if (t->nit >= t->maxiter) return LP_LIMIT;
        const int bland = degenerate > 50;
        int64_t q = -1;
        double best = -t->tol;
        for (int64_t j = 0; j < w; j++) {
            if (t->banned[j]) continue;
            double d = AT(t, m, j);
            if (d < best) {
                q = j;
                if (bland) break;
                best = d;
            }
        }
        if (q < 0) return LP_OPTIMAL;
        /* Harris ratio test: pass 1 with relaxed bounds, pass 2 picks the largest pivot within it */
        const double ptol = 1e-11;
        double theta = INFINITY;
        for (int64_t i = 0; i < m; i++) {
            double a = AT(t, i, q);
            if (a > ptol) {
                double r = (AT(t, i, w) + t->tol) / a;
                if (r < theta) theta = r;
            }
        }
        if (theta == INFINITY) return LP_UNBOUNDED;
        int64_t r = -1;
        double big = 0.0;
        for (int64_t i = 0; i < m; i++) {
            double a = AT(t, i, q);
            if (a > ptol && AT(t, i, w) / a <= theta) {
                if (bland ? (r < 0 || t->basis[i] < t->basis[r]) : a > big) {
                    big = a;
                    r = i;
                }
            }
        }
        if (r < 0) return LP_UNBOUNDED;
        degenerate = AT(t, r, w) <= t->tol ? degenerate + 1 : 0;
        pivot(t, r, q);
        /* keep the rhs non-negative against round-off */
        for (int64_t i = 0; i < m; i++)
            if (AT(t, i, w) < 0 && AT(t, i, w) > -t->tol) AT(t, i, w) = 0.0;
    }
}

/* Objective row for costs `cost` (length w) given the current basis. */
static void set_objective(tab_t *t, const double *cost)
{
    const int64_t m = t->m, w = t->w;
    for (int64_t j = 0; j <= w; j++) AT(t, m, j) = j < w ? cost[j] : 0.0;
    for (int64_t i = 0; i < m; i++) {
        double cb = cost[t->basis[i]];
        if (cb == 0.0) continue;
        for (int64_t j = 0; j <= w; j++) AT(t, m, j) -= cb * AT(t, i, j);
    }
}

/* SciPy rejects NaN/inf in c, A and b (and NaN in bounds): so do we, before any work. */
static int lp_inputs_ok(int64_t n, const double *c, int64_t m_ub, const double *A_ub, const double *b_ub,
                        int64_t m_eq, const double *A_eq, const double *b_eq, const double *lb, const double *ub)
{
    for (int64_t j = 0; j < n; j++) {
        if (!isfinite(c[j])) return 0;
        if ((lb && isnan(lb[j])) || (ub && isnan(ub[j]))) return 0;
    }
    for (int64_t k = 0; k < m_ub * n; k++) if (!isfinite(A_ub[k])) return 0;
    for (int64_t k = 0; k < m_ub; k++) if (!isfinite(b_ub[k])) return 0;
    for (int64_t k = 0; k < m_eq * n; k++) if (!isfinite(A_eq[k])) return 0;
    for (int64_t k = 0; k < m_eq; k++) if (!isfinite(b_eq[k])) return 0;
    return 1;
}

int tsr_linprog(int64_t n, const double *c,
                int64_t m_ub, const double *A_ub, const double *b_ub,
                int64_t m_eq, const double *A_eq, const double *b_eq,
                const double *lb, const double *ub, int64_t maxiter, double tol,
                double *x, double *fun, double *duals_ub, double *duals_eq, double *reduced, int64_t *nit)
{
    if (n <= 0 || m_ub < 0 || m_eq < 0 || !c || !x || !fun) return TSR_EARG;
    if ((m_ub > 0 && (!A_ub || !b_ub)) || (m_eq > 0 && (!A_eq || !b_eq))) return TSR_EARG;
    if (!lp_inputs_ok(n, c, m_ub, A_ub, b_ub, m_eq, A_eq, b_eq, lb, ub)) return TSR_EARG;
    if (maxiter <= 0) maxiter = 100000;
    if (tol <= 0) tol = 1e-9;

    /* ---- variable substitution ---- */
    int *kind = (int *)malloc((size_t)n * sizeof(int));
    int64_t *col = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    double *shift = (double *)malloc((size_t)n * sizeof(double));
    if (!kind || !col || !shift) { free(kind); free(col); free(shift); return TSR_ENOMEM; }
    int64_t ns = 0, m_bnd = 0;
    for (int64_t j = 0; j < n; j++) {
        double lo = lb ? lb[j] : 0.0, hi = ub ? ub[j] : INFINITY;
        if (lo > hi) { free(kind); free(col); free(shift); return LP_INFEASIBLE; }
        col[j] = ns;
        if (isfinite(lo)) {
            kind[j] = V_SHIFT;
            shift[j] = lo;
            ns++;
            if (isfinite(hi)) m_bnd++;
        } else if (isfinite(hi)) {
            kind[j] = V_MIRROR;
            shift[j] = hi;
            ns++;
        } else {
            kind[j] = V_FREE;
            shift[j] = 0.0;
            ns += 2;
        }
    }
    const int64_t m = m_ub + m_bnd + m_eq;
    const int64_t n_slack = m_ub + m_bnd;

    /* rows in the substituted space: coef (m x ns), rhs (m) */
    /* the two O(m * n) blocks go through tsr_calloc so they count against the memory budget */
    const int64_t coef_bytes = (m > 0 ? m : 1) * ns * (int64_t)sizeof(double);
    double *coef = (double *)tsr_calloc(coef_bytes);
    double *rhs = (double *)calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    int *flip = (int *)calloc((size_t)(m > 0 ? m : 1), sizeof(int));
    if (!coef || !rhs || !flip) { free(kind); free(col); free(shift); tsr_free(coef, coef_bytes); free(rhs); free(flip); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++) {
        const double *arow = NULL;
        double b;
        if (i < m_ub) { arow = A_ub + i * n; b = b_ub[i]; }
        else if (i < m_ub + m_bnd) { b = 0.0; }
        else { arow = A_eq + (i - m_ub - m_bnd) * n; b = b_eq[i - m_ub - m_bnd]; }
        double *row = coef + i * ns;
        if (arow) {
            for (int64_t j = 0; j < n; j++) {
                double a = arow[j];
                if (a == 0.0) continue;
                switch (kind[j]) {
                case V_SHIFT: row[col[j]] += a; b -= a * shift[j]; break;
                case V_MIRROR: row[col[j]] -= a; b -= a * shift[j]; break;
                default: row[col[j]] += a; row[col[j] + 1] -= a; break;
                }
            }
        }
        rhs[i] = b;
    }
    /* bound rows: y_j <= ub_j - lb_j */
    for (int64_t j = 0, k = m_ub; j < n; j++) {
        if (kind[j] == V_SHIFT && ub && isfinite(ub[j])) {
            coef[k * ns + col[j]] = 1.0;
            rhs[k] = ub[j] - lb[j];
            k++;
        }
    }
    for (int64_t i = 0; i < m; i++) {
        if (rhs[i] < 0) {
            flip[i] = 1;
            rhs[i] = -rhs[i];
            for (int64_t j = 0; j < ns; j++) coef[i * ns + j] = -coef[i * ns + j];
        }
    }
    /* columns: structural | slacks | artificials */
    int64_t n_art = 0;
    for (int64_t i = 0; i < m; i++)
        if (i >= n_slack || flip[i]) n_art++;
    tab_t t;
    t.m = m;
    t.w = ns + n_slack + n_art;
    const int64_t tab_bytes = (m + 1) * (t.w + 1) * (int64_t)sizeof(double);
    t.T = (double *)tsr_calloc(tab_bytes);
    t.basis = (int64_t *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int64_t));
    t.banned = (uint8_t *)calloc((size_t)t.w, 1);
    int64_t *idcol = (int64_t *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int64_t));
    double *cost = (double *)calloc((size_t)t.w, sizeof(double));
    t.nit = 0;
    t.maxiter = maxiter;
    t.tol = tol;
    int status = LP_NUMERICAL;
    if (!t.T || !t.basis || !t.banned || !idcol || !cost) { status = TSR_ENOMEM; goto done; }

    {
        int64_t art = ns + n_slack;
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < ns; j++) AT(&t, i, j) = coef[i * ns + j];
            AT(&t, i, t.w) = rhs[i];
            if (i < n_slack) AT(&t, i, ns + i) = flip[i] ? -1.0 : 1.0;
            if (i >= n_slack || flip[i]) {
                AT(&t, i, art) = 1.0;
                t.basis[i] = art;
                idcol[i] = art;
                art++;
            } else {
                t.basis[i] = ns + i;
                idcol[i] = ns + i;
            }
        }
    }

    /* ---- phase 1 ---- */
    if (n_art > 0) {
        for (int64_t j = ns + n_slack; j < t.w; j++) cost[j] = 1.0;
        for (int64_t j = ns + n_slack; j < t.w; j++) t.banned[j] = 1;
        set_objective(&t, cost);
        int r = simplex(&t);
        if (r == LP_LIMIT) { status = LP_LIMIT; goto done; }
        double scale = 1.0;
        for (int64_t i = 0; i < m; i++) scale = fmax(scale, rhs[i]);
        if (-AT(&t, m, t.w) > 1e3 * tol * scale) { status = LP_INFEASIBLE; goto done; }
        /* drive remaining (zero-level) artificials out of the basis */
        for (int64_t i = 0; i < m; i++) {
            if (t.basis[i] < ns + n_slack) continue;
            int64_t q = -1;
            double big = 1e-9;
            for (int64_t j = 0; j < ns + n_slack; j++) {
                if (fabs(AT(&t, i, j)) > big) { big = fabs(AT(&t, i, j)); q = j; }
            }
            if (q >= 0) pivot(&t, i, q);
        }
        memset(cost, 0, (size_t)t.w * sizeof(double));
    }

    /* ---- phase 2 ---- */
    for (int64_t j = 0; j < n; j++) {
        switch (kind[j]) {
        case V_SHIFT: cost[col[j]] = c[j]; break;
        case V_MIRROR: cost[col[j]] = -c[j]; break;
        default: cost[col[j]] = c[j]; cost[col[j] + 1] = -c[j]; break;
        }
    }
    set_objective(&t, cost);
    status = simplex(&t);
    if (status == LP_OPTIMAL || status == LP_LIMIT) {
        double *y = (double *)calloc((size_t)ns, sizeof(double));
        if (!y) { status = TSR_ENOMEM; goto done; }
        for (int64_t i = 0; i < m; i++)
            if (t.basis[i] < ns) y[t.basis[i]] = AT(&t, i, t.w);
        double obj = 0.0;
        for (int64_t j = 0; j < n; j++) {
            double v;
            switch (kind[j]) {
            case V_SHIFT: v = shift[j] + y[col[j]]; break;
            case V_MIRROR: v = shift[j] - y[col[j]]; break;
            default: v = y[col[j]] - y[col[j] + 1]; break;
            }
            x[j] = v;
            obj += c[j] * v;
            if (reduced) {
                double d = AT(&t, m, col[j]);
                reduced[j] = kind[j] == V_MIRROR ? -d : d;
            }
        }
        *fun = obj;
        free(y);
        for (int64_t i = 0; i < m; i++) {
            double yi = -AT(&t, m, idcol[i]);
            if (flip[i]) yi = -yi;
            if (i < m_ub) { if (duals_ub) duals_ub[i] = yi; }
            else if (i >= n_slack) { if (duals_eq) duals_eq[i - n_slack] = yi; }
        }
    }
done:
    if (nit) *nit = t.nit;
    free(kind); free(col); free(shift); tsr_free(coef, coef_bytes); free(rhs); free(flip);
    tsr_free(t.T, tab_bytes); free(t.basis); free(t.banned); free(idcol); free(cost);
    return status;
}

/* ------------------------------------------------------------------ MILP */

typedef struct { double *lb, *ub; double bound; } node_t;

int tsr_milp(int64_t n, const double *c,
             int64_t m_ub, const double *A_ub, const double *b_ub,
             int64_t m_eq, const double *A_eq, const double *b_eq,
             const double *lb, const double *ub, const uint8_t *integrality,
             int64_t node_limit, double mip_rel_gap, double tol,
             double *x, double *fun, int64_t *nodes_out, double *best_bound)
{
    if (n <= 0 || m_ub < 0 || m_eq < 0 || !c || !x || !fun) return TSR_EARG;
    if ((m_ub > 0 && (!A_ub || !b_ub)) || (m_eq > 0 && (!A_eq || !b_eq))) return TSR_EARG;
    if (!lp_inputs_ok(n, c, m_ub, A_ub, b_ub, m_eq, A_eq, b_eq, lb, ub)) return TSR_EARG;
    if (node_limit <= 0) node_limit = 100000;
    if (tol <= 0) tol = 1e-9;
    const double int_tol = 1e-6;
    int64_t cap = 64, top = 0, nodes = 0;
    node_t *stack = (node_t *)malloc((size_t)cap * sizeof(node_t));
    double *xr = (double *)malloc((size_t)n * sizeof(double));
    if (!stack || !xr) { free(stack); free(xr); return TSR_ENOMEM; }
    node_t root = {(double *)malloc((size_t)n * sizeof(double)), (double *)malloc((size_t)n * sizeof(double)), -INFINITY};
    if (!root.lb || !root.ub) { free(stack); free(xr); free(root.lb); free(root.ub); return TSR_ENOMEM; }
    for (int64_t j = 0; j < n; j++) {
        root.lb[j] = lb ? lb[j] : 0.0;
        root.ub[j] = ub ? ub[j] : INFINITY;
        if (integrality && integrality[j]) {
            if (isfinite(root.lb[j])) root.lb[j] = ceil(root.lb[j] - int_tol);
            if (isfinite(root.ub[j])) root.ub[j] = floor(root.ub[j] + int_tol);
        }
    }
    stack[top++] = root;
    double incumbent = INFINITY;
    int have = 0, status = LP_INFEASIBLE, unbounded = 0, limit = 0;
    double open_bound = INFINITY;

    while (top > 0) {
        node_t nd = stack[--top];
        if (nodes >= node_limit) {
            limit = 1;
            open_bound = fmin(open_bound, nd.bound);
            free(nd.lb); free(nd.ub);
            continue;
        }
        if (have && nd.bound >= incumbent - mip_rel_gap * fabs(incumbent) - tol) { free(nd.lb); free(nd.ub); continue; }
        nodes++;
        double f;
        int64_t it;
        int r = tsr_linprog(n, c, m_ub, A_ub, b_ub, m_eq, A_eq, b_eq, nd.lb, nd.ub, 0, tol, xr, &f, NULL, NULL, NULL, &it);
        if (r < 0) { free(nd.lb); free(nd.ub); status = r; goto out; }   /* out of memory / budget: report, never prune */
        if (r == LP_UNBOUNDED) { unbounded = 1; free(nd.lb); free(nd.ub); break; }
        if (r == LP_LIMIT || r == LP_NUMERICAL) {
            /* the relaxation was not solved: the subtree is unexplored, so optimality cannot be claimed */
            limit = 1;
            open_bound = fmin(open_bound, nd.bound);
            free(nd.lb); free(nd.ub);
            continue;
        }
        if (r != LP_OPTIMAL || (have && f >= incumbent - mip_rel_gap * fabs(incumbent) - tol)) { free(nd.lb); free(nd.ub); continue; }
        /* most fractional integer variable */
        int64_t bj = -1;
        double bfrac = 0.0;
        for (int64_t j = 0; j < n; j++) {
            if (!integrality || !integrality[j]) continue;
            double fr = xr[j] - floor(xr[j]);
            double dist = fmin(fr, 1.0 - fr);
            if (dist > int_tol && dist > bfrac) { bfrac = dist; bj = j; }
        }
        if (bj < 0) {
            incumbent = f;
            have = 1;
            for (int64_t j = 0; j < n; j++) x[j] = (integrality && integrality[j]) ? round(xr[j]) : xr[j];
            free(nd.lb); free(nd.ub);
            continue;
        }
        if (top + 2 > cap) {
            cap *= 2;
            node_t *ns = (node_t *)realloc(stack, (size_t)cap * sizeof(node_t));
            if (!ns) { free(nd.lb); free(nd.ub); status = TSR_ENOMEM; goto out; }
            stack = ns;
        }
        node_t down = {(double *)malloc((size_t)n * sizeof(double)), (double *)malloc((size_t)n * sizeof(double)), f};
        node_t up = {nd.lb, nd.ub, f};
        if (!down.lb || !down.ub) { free(down.lb); free(down.ub); free(nd.lb); free(nd.ub); status = TSR_ENOMEM; goto out; }
        memcpy(down.lb, nd.lb, (size_t)n * sizeof(double));
        memcpy(down.ub, nd.ub, (size_t)n * sizeof(double));
        down.ub[bj] = floor(xr[bj]);
        up.lb[bj] = ceil(xr[bj]);
        /* explore the nearer rounding first (it is pushed last) */
        if (xr[bj] - floor(xr[bj]) < 0.5) { stack[top++] = up; stack[top++] = down; }
        else { stack[top++] = down; stack[top++] = up; }
    }
    if (unbounded && !have) status = LP_UNBOUNDED;
    else if (have) status = limit ? LP_LIMIT : LP_OPTIMAL;
    else status = limit ? LP_LIMIT : LP_INFEASIBLE;
    if (have) {
        double s = 0.0;
        for (int64_t j = 0; j < n; j++) s += c[j] * x[j];
        *fun = s;
    }
    if (best_bound) *best_bound = have && !limit ? *fun : fmin(open_bound, have ? *fun : INFINITY);
out:
    while (top > 0) { top--; free(stack[top].lb); free(stack[top].ub); }
    free(stack);
    free(xr);
    if (nodes_out) *nodes_out = nodes;
    return status;
}
