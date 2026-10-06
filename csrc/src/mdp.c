/*
 * Markov decision processes (discounted, finite state and action sets).
 *
 * Transitions are stored as one CSR matrix with A*S rows: row a*S + s holds
 * P(. | s, a). Dense (A, S, S) tensors are converted once by the caller
 * (tsr_dense_to_csr), so every solver has one code path and sparse models
 * with millions of states cost memory proportional to their non-zeros.
 * Rewards R are dense (S, A): expected immediate reward of action a in s.
 *
 *   tsr_mdp_bellman          one Bellman backup: Q, greedy V and policy
 *   tsr_mdp_value_iteration  successive approximation to an eps-optimal policy
 *   tsr_mdp_policy_iteration Howard's policy iteration; exact evaluation by LU
 *                            (eval_sweeps = 0) or modified PI with k sweeps
 *   tsr_mdp_policy_eval      value of a fixed policy
 *   tsr_mdp_finite_horizon   backward induction over T stages
 *
 * Ties pick the lowest action index (numpy.argmax). The per-state loops run
 * in parallel in OpenMP builds; results do not depend on the thread count.
 */
#include "internal.h"
#include <math.h>
#include <stdlib.h>

/* Q(s, a) = R(s, a) + gamma * sum_s' P(s'|s,a) V(s'); returns max over a in V_out, argmax in pol. */
static void backup(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                   const double *R, double gamma, const double *V, double *V_out, int64_t *pol, double *Q)
{
    const int threads = tsr_get_threads();
    (void)threads;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1 && S * A > 4096)
#endif
    for (int64_t s = 0; s < S; s++) {
        double best = -INFINITY;
        int64_t arg = 0;
        for (int64_t a = 0; a < A; a++) {
            const int64_t row = a * S + s;
            double ev = 0.0;
            for (int64_t p = indptr[row]; p < indptr[row + 1]; p++) ev += data[p] * V[indices[p]];
            const double q = R[s * A + a] + gamma * ev;
            if (Q) Q[s * A + a] = q;
            if (q > best) { best = q; arg = a; }
        }
        V_out[s] = best;
        if (pol) pol[s] = arg;
    }
}

static int check_model(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                       const double *R, double gamma)
{
    if (S <= 0 || A <= 0 || !indptr || !R || gamma < 0.0 || gamma > 1.0) return TSR_EARG;
    if (indptr[A * S] > 0 && (!indices || !data)) return TSR_EARG;
    for (int64_t r = 0; r < A * S; r++) {
        if (indptr[r + 1] < indptr[r]) return TSR_EARG;
        for (int64_t p = indptr[r]; p < indptr[r + 1]; p++)
            if (indices[p] < 0 || indices[p] >= S) return TSR_EINDEX;
    }
    return TSR_OK;
}

int tsr_mdp_bellman(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                    const double *R, double gamma, const double *V, double *Q, double *V_out, int64_t *policy)
{
    int rc = check_model(S, A, indptr, indices, data, R, gamma);
    if (rc != TSR_OK) return rc;
    backup(S, A, indptr, indices, data, R, gamma, V, V_out, policy, Q);
    return TSR_OK;
}

/*
 * Value iteration from V (in: initial values, out: final values). Stops when
 * the sup-norm change drops below epsilon * (1 - gamma) / (2 * gamma), which
 * makes the greedy policy epsilon-optimal (Puterman 6.3.1); for gamma = 1 the
 * span of the change is used instead. Returns 0 converged, 1 max_iter reached.
 */
int tsr_mdp_value_iteration(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                            const double *R, double gamma, double epsilon, int64_t max_iter,
                            double *V, int64_t *policy, int64_t *iterations, double *delta_out)
{
    int rc = check_model(S, A, indptr, indices, data, R, gamma);
    if (rc != TSR_OK) return rc;
    if (epsilon <= 0) epsilon = 1e-6;
    if (max_iter <= 0) max_iter = 10000;
    double *Vn = (double *)tsr_alloc(S * 8);
    if (!Vn) return TSR_ENOMEM;
    const double thresh = gamma < 1.0 ? (gamma > 0.0 ? epsilon * (1.0 - gamma) / (2.0 * gamma) : INFINITY) : epsilon;
    int64_t it = 0;
    double delta = INFINITY;
    int status = 1;
    while (it < max_iter) {
        backup(S, A, indptr, indices, data, R, gamma, V, Vn, policy, NULL);
        it++;
        double dmax = -INFINITY, dmin = INFINITY, sup = 0.0;
        for (int64_t s = 0; s < S; s++) {
            double d = Vn[s] - V[s];
            dmax = fmax(dmax, d);
            dmin = fmin(dmin, d);
            sup = fmax(sup, fabs(d));
        }
        memcpy(V, Vn, (size_t)S * 8);
        delta = gamma < 1.0 ? sup : dmax - dmin;
        if (delta < thresh) { status = 0; break; }
    }
    /* greedy policy with respect to the returned V */
    backup(S, A, indptr, indices, data, R, gamma, V, Vn, policy, NULL);
    tsr_free(Vn, S * 8);
    if (iterations) *iterations = it;
    if (delta_out) *delta_out = delta;
    return status;
}

/* Solve (I - gamma P_pi) V = R_pi by LU with partial pivoting (dense, S x S). */
static int eval_exact(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                      const double *R, double gamma, const int64_t *pol, double *V)
{
    double *M = (double *)tsr_calloc(S * S * 8);
    if (!M) return TSR_ENOMEM;
    for (int64_t s = 0; s < S; s++) {
        const int64_t row = pol[s] * S + s;
        M[s * S + s] = 1.0;
        for (int64_t p = indptr[row]; p < indptr[row + 1]; p++) M[s * S + indices[p]] -= gamma * data[p];
        V[s] = R[s * A + pol[s]];
    }
    for (int64_t k = 0; k < S; k++) {
        int64_t piv = k;
        double big = fabs(M[k * S + k]);
        for (int64_t i = k + 1; i < S; i++)
            if (fabs(M[i * S + k]) > big) { big = fabs(M[i * S + k]); piv = i; }
        if (big < 1e-300) { tsr_free(M, S * S * 8); return TSR_ECONVERGE; } /* singular: gamma = 1 with a recurrent class */
        if (piv != k) {
            for (int64_t j = 0; j < S; j++) { double t = M[k * S + j]; M[k * S + j] = M[piv * S + j]; M[piv * S + j] = t; }
            double t = V[k]; V[k] = V[piv]; V[piv] = t;
        }
        const double d = M[k * S + k];
        for (int64_t i = k + 1; i < S; i++) {
            const double f = M[i * S + k] / d;
            if (f == 0.0) continue;
            double *ri = M + i * S;
            const double *rk = M + k * S;
            for (int64_t j = k; j < S; j++) ri[j] -= f * rk[j];
            V[i] -= f * V[k];
        }
    }
    for (int64_t i = S - 1; i >= 0; i--) {
        double s = V[i];
        for (int64_t j = i + 1; j < S; j++) s -= M[i * S + j] * V[j];
        V[i] = s / M[i * S + i];
    }
    tsr_free(M, S * S * 8);
    return TSR_OK;
}

/* k sweeps of V <- R_pi + gamma P_pi V (modified policy iteration's partial evaluation). */
static int eval_sweeps(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                       const double *R, double gamma, const int64_t *pol, double *V, int64_t k)
{
    double *Vn = (double *)tsr_alloc(S * 8);
    if (!Vn) return TSR_ENOMEM;
    for (int64_t sweep = 0; sweep < k; sweep++) {
        for (int64_t s = 0; s < S; s++) {
            const int64_t row = pol[s] * S + s;
            double ev = 0.0;
            for (int64_t p = indptr[row]; p < indptr[row + 1]; p++) ev += data[p] * V[indices[p]];
            Vn[s] = R[s * A + pol[s]] + gamma * ev;
        }
        memcpy(V, Vn, (size_t)S * 8);
    }
    tsr_free(Vn, S * 8);
    return TSR_OK;
}

int tsr_mdp_policy_eval(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                        const double *R, double gamma, const int64_t *policy, double *V)
{
    int rc = check_model(S, A, indptr, indices, data, R, gamma);
    if (rc != TSR_OK) return rc;
    for (int64_t s = 0; s < S; s++)
        if (policy[s] < 0 || policy[s] >= A) return TSR_EINDEX;
    return eval_exact(S, A, indptr, indices, data, R, gamma, policy, V);
}

/*
 * Policy iteration from `policy` (in: initial policy, out: optimal policy).
 * eval_sweeps = 0 evaluates exactly (LU, O(S^3)); k > 0 runs modified policy
 * iteration with k evaluation sweeps and stops like value iteration.
 * Returns 0 when the policy is stable, 1 at max_iter.
 */
int tsr_mdp_policy_iteration(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                             const double *R, double gamma, int64_t eval_sweeps_k, double epsilon, int64_t max_iter,
                             double *V, int64_t *policy, int64_t *iterations)
{
    int rc = check_model(S, A, indptr, indices, data, R, gamma);
    if (rc != TSR_OK) return rc;
    if (max_iter <= 0) max_iter = 1000;
    if (epsilon <= 0) epsilon = 1e-6;
    for (int64_t s = 0; s < S; s++)
        if (policy[s] < 0 || policy[s] >= A) policy[s] = 0;
    double *Vg = (double *)tsr_alloc(S * 8);
    int64_t *np = (int64_t *)tsr_alloc(S * 8);
    if (!Vg || !np) { if (Vg) tsr_free(Vg, S * 8); if (np) tsr_free(np, S * 8); return TSR_ENOMEM; }
    int status = 1;
    int64_t it = 0;
    if (eval_sweeps_k > 0) memset(V, 0, (size_t)S * 8);
    while (it < max_iter) {
        it++;
        if (eval_sweeps_k == 0) {
            rc = eval_exact(S, A, indptr, indices, data, R, gamma, policy, V);
        } else {
            rc = eval_sweeps(S, A, indptr, indices, data, R, gamma, policy, V, eval_sweeps_k);
        }
        if (rc != TSR_OK) { status = rc; break; }
        backup(S, A, indptr, indices, data, R, gamma, V, Vg, np, NULL);
        if (eval_sweeps_k == 0) {
            /* keep the incumbent action unless another is strictly better (avoids flip-flopping on ties) */
            int changed = 0;
            for (int64_t s = 0; s < S; s++) {
                const int64_t row = policy[s] * S + s;
                double ev = 0.0;
                for (int64_t p = indptr[row]; p < indptr[row + 1]; p++) ev += data[p] * V[indices[p]];
                double cur = R[s * A + policy[s]] + gamma * ev;
                if (np[s] != policy[s] && Vg[s] > cur + 1e-12 * (1.0 + fabs(cur))) { policy[s] = np[s]; changed = 1; }
            }
            if (!changed) { status = 0; break; }
        } else {
            double sup = 0.0;
            for (int64_t s = 0; s < S; s++) sup = fmax(sup, fabs(Vg[s] - V[s]));
            memcpy(policy, np, (size_t)S * 8);
            memcpy(V, Vg, (size_t)S * 8);
            if (gamma < 1.0 && sup < epsilon * (1.0 - gamma) / (2.0 * gamma)) { status = 0; break; }
        }
    }
    tsr_free(Vg, S * 8);
    tsr_free(np, S * 8);
    if (iterations) *iterations = it;
    return status;
}

/*
 * Backward induction over T stages. V_terminal (S) may be NULL (zeros).
 * V_out: (T + 1) x S; row t is the optimal value at the start of stage t
 * (T - t decisions left), row T the terminal values.
 * policy_out: T x S.
 */
int tsr_mdp_finite_horizon(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                           const double *R, double gamma, int64_t T, const double *V_terminal,
                           double *V_out, int64_t *policy_out)
{
    int rc = check_model(S, A, indptr, indices, data, R, gamma);
    if (rc != TSR_OK) return rc;
    if (T <= 0 || !V_out || !policy_out) return TSR_EARG;
    double *VT = V_out + T * S;
    if (V_terminal) memcpy(VT, V_terminal, (size_t)S * 8);
    else memset(VT, 0, (size_t)S * 8);
    for (int64_t t = T - 1; t >= 0; t--)
        backup(S, A, indptr, indices, data, R, gamma, V_out + (t + 1) * S, V_out + t * S, policy_out + t * S, NULL);
    return TSR_OK;
}
