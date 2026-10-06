# LinearProgramming

`Tessero\Optimize\LinearProgramming`

scipy.optimize.linprog and scipy.optimize.milp on libtessero's native
simplex and branch-and-bound.

  minimise c.x  subject to  A_ub x <= b_ub,  A_eq x = b_eq,  bounds

Bounds follow linprog: null means (0, +inf) for every variable; one
[lo, hi] pair applies to all; otherwise one pair per variable. null or
+/-INF is an open side. Maximise by negating c (the result's fun is then
the negated maximum, as in SciPy).

## Constants

| Name | Value |
|---|---|
| `STATUS` | `["Optimization terminated successfully.","Iteration or node limit reached.","The problem is infeasible.","The problem is unbounded.","Numerical difficulties encountered."]` |

## Methods

### linprog

```php
static linprog(mixed $c, mixed $A_ub = null, mixed $b_ub = null, mixed $A_eq = null, mixed $b_eq = null, ?array $bounds = null, int $maxiter = 100000, float $tol = 1.0E-9): Tessero\Optimize\LinprogResult
```

### milp

```php
static milp(mixed $c, array|bool $integrality = true, mixed $A_ub = null, mixed $b_ub = null, mixed $A_eq = null, mixed $b_eq = null, ?array $bounds = null, int $nodeLimit = 100000, float $mipRelGap = 0.0, float $tol = 1.0E-9): Tessero\Optimize\MilpResult
```

Mixed-integer linear program (scipy.optimize.milp). $integrality: one flag per
variable (true/1 = integer), or a single bool for all.
