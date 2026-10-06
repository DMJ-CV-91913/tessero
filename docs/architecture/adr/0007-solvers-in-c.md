# 0007. LP/MILP and MDP solvers in C

- Status: Accepted
- Date: 2026-09-26

## Context

Pure-PHP simplex and value-iteration loops were slow, and they were exposed to
PHP JIT defects: the tracing JIT miscompiled a simplex inner loop. Planning and
decision problems (production planning, dispatch, maintenance policies,
inventory) are a core use case. Dedicated solvers (HiGHS, GLPK) exist but add
native dependencies.

## Decision

- `tsr_linprog`: dense two-phase primal simplex with Harris's ratio test,
  Bland's rule after degenerate runs, and SciPy-compatible duals.
  `tsr_milp`: depth-first branch and bound on it. No external dependency.
- MDPs store transitions as one CSR matrix with A·S rows, so dense and sparse
  models share every solver. Value iteration uses an ε-optimal stopping rule;
  policy iteration is exact (LU) or modified.
- Inputs are validated in C (non-finite data, malformed CSR). Workspaces count
  against the memory budget.

## Consequences

- One call per solve, with no per-iteration PHP overhead and no exposure to the JIT.
- The dense tableau limits LP size to the low thousands of rows and columns.
  MILP has no cuts or presolve. Both limits are documented, and an optional
  HiGHS backend is on the roadmap for large models.
- Results follow SciPy's conventions, so models ported from Python behave the same.
