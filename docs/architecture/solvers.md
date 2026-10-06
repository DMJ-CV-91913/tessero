# Solvers

All solvers run their iteration loops in C. PHP builds the inputs, makes one
call, and wraps the result. User-supplied callables (the `minimize`/`root`
family) are the exception, since the objective is PHP code.

## Linear programming (`tsr_linprog`)

**Problem form:** minimise cᵀx subject to A_ub x ≤ b_ub, A_eq x = b_eq,
lb ≤ x ≤ ub.

**Pre-processing**

1. Inputs are validated: finite `c`, `A`, `b`; bounds may be infinite but not NaN.
2. Bounds are removed by substitution. A variable with a finite lower bound
   is shifted (x = lb + y). One with only an upper bound is mirrored
   (x = ub − y). A free variable is split (x = y⁺ − y⁻). Finite upper bounds of
   shifted variables become extra ≤ rows.
3. Rows with negative right-hand side are negated (and remembered), so the
   initial basis has b ≥ 0.
4. Slack columns are added for ≤ rows. Artificial columns are added for
   equality rows and negated ≤ rows.

**Phase 1** minimises the sum of artificials. If the optimum is above
1e3 · tol · scale, the problem is infeasible (status 2). Artificials still
basic at zero are pivoted out where possible, and banned from re-entering.

**Phase 2** minimises the real objective from the phase 1 basis.

**Pivoting**

- Entering column: Dantzig's rule (most negative reduced cost). After a run of
  degenerate pivots it switches to Bland's rule (lowest index), which cannot
  cycle.
- Leaving row: Harris's two-pass ratio test. Pass 1 finds the smallest ratio
  with a tolerance-relaxed bound. Pass 2 picks, among rows within it, the
  largest pivot element. This avoids tiny pivots and the error growth they
  cause.
- A column with no positive entry means the problem is unbounded (status 3).

**Outputs:** x (mapped back through the substitutions), the objective, row
duals (the reduced costs of slack/artificial columns, sign-adjusted for
negated rows: SciPy's `marginals` convention), reduced costs and the iteration
count.

**Complexity:** dense tableau of (m+1) × (n_s+m+1) doubles, counted against
the memory budget. Each pivot is O(m · n). Suitable for hundreds to a few
thousand rows and columns.

**Verification:** 14 LPs checked against SciPy's HiGHS: optimal, unbounded,
infeasible, mixed equalities, free and bounded variables, degenerate vertices.
Objectives match to 1e-9, and duals match wherever they are unique.

## Mixed-integer programming (`tsr_milp`)

Depth-first branch and bound over `tsr_linprog`:

- Each node is a pair of bound vectors. The root rounds integer variables'
  bounds inward.
- Branch on the most fractional integer variable (tolerance 1e-6). Two
  children get `x ≤ ⌊v⌋` and `x ≥ ⌈v⌉`. The child nearer the relaxation value
  is explored first (pushed last).
- Prune when the node bound is no better than the incumbent (within
  `mip_rel_gap`) or the relaxation is infeasible.
- A relaxation that stops on its iteration limit or with numerical trouble is
  **not** pruned silently. The subtree is recorded as unexplored, the result
  status becomes 1 (limit), and `best_bound` includes it, so `gap()` stays
  honest.
- Out-of-memory/budget in a relaxation stops the search with `TSR_ENOMEM`.

Checked against `scipy.optimize.milp` on 10 problems (knapsack, assignment,
facility location, mixed continuous/integer, infeasible, bounded integer
ranges).

No cuts, presolve or heuristics. Performance depends on the LP bound's
tightness. See the [roadmap](../project/roadmap.md) for HiGHS integration.

## Markov decision processes (`mdp.c`)

**Model representation:** one CSR matrix with A·S rows; row a·S + s is
P(· | s, a). R is dense (S × A). Memory is O(nnz + S·A).

**Bellman backup** (`backup`): for each state, and each action,
Q(s,a) = R(s,a) + γ Σ P(s′|s,a) V(s′). Take the max and argmax (lowest index on
ties). States are independent, so the loop over states is parallel (OpenMP
static schedule) above 4 096 state-action pairs.

| Solver | Method | Stopping rule |
|---|---|---|
| Value iteration | V ← max_a Q | sup-norm change < ε(1−γ)/(2γ), which guarantees an ε-optimal greedy policy (Puterman, Theorem 6.3.1); span of the change for γ = 1 |
| Policy iteration | evaluate π exactly (dense LU of I − γP_π, partial pivoting), improve greedily | policy unchanged |
| Modified PI | k sweeps of V ← R_π + γP_π V instead of exact evaluation | as value iteration |
| Policy evaluation | dense LU | — (singular I − γP_π, e.g. γ = 1 with a recurrent class, returns −8) |
| Finite horizon | backward induction from terminal values for T stages | exact |

The greedy policy is always recomputed from the final values, so
the returned policy is consistent with the returned values.

Checked against NumPy reference implementations on 4 models (dense and sparse,
γ < 1, finite horizon).

## Sparse iterative solvers (`sparse.c`)

- **CG** for symmetric positive definite A, and **BiCGSTAB** for general
  square A, both with a Jacobi (diagonal) preconditioner.
- Stopping rule: ‖r‖ ≤ max(rtol·‖b‖, atol), SciPy's convention. A breakdown
  (ρ = 0 or ω = 0 in BiCGSTAB) returns a non-zero `info`.
- Checked against `scipy.sparse.linalg` on SPD and non-symmetric systems.

## Optimisation with PHP callables (`src/Optimize`)

| Method | Algorithm | Fidelity to SciPy |
|---|---|---|
| BFGS | quasi-Newton with SciPy's Wolfe line search (`dcsrch` port) | same iterates on the test problems |
| L-BFGS-B | projected limited-memory BFGS with bounds | same optimum, different path (not a port of the Fortran code) |
| Nelder-Mead | SciPy's simplex rules, including `adaptive` | same iterates |
| brentq | Brent's method as in SciPy's C code | same root and call count |
| newton / secant | SciPy's `newton` | same result |
| root (systems) | damped Newton with a finite-difference Jacobian | converges on the test systems |
| least squares, curve_fit | Levenberg–Marquardt; covariance scaled as SciPy | parameters and covariance to tolerance |

These run in PHP because they call back into PHP every iteration. Their
linear algebra (solving the Newton or LM step) is native.
