# Linear and mixed-integer programming

`Tessero\Optimize\LinearProgramming` solves

```
minimise    c · x
subject to  A_ub x ≤ b_ub
            A_eq x = b_eq
            lb ≤ x ≤ ub
            (optionally) some x integer
```

with the conventions of `scipy.optimize.linprog` and `scipy.optimize.milp`.
The solve runs entirely in C: one PHP call per problem.

## linprog

```php
use Tessero\Optimize\LinearProgramming as LP;

// Two products share two resources.
// Product A earns 40 and uses 2 h machining + 1 h assembly;
// product B earns 30 and uses 1 h + 1 h. 100 h machining, 80 h assembly.
$r = LP::linprog(
    c:    [-40, -30],                  // maximise profit = minimise its negative
    A_ub: [[2, 1], [1, 1]],
    b_ub: [100, 80],
);

$r->success;            // true
$r->x;                  // [20.0, 60.0]
$r->fun;                // -2600.0  (profit 2600)
$r->ineqlinMarginals;   // [-10.0, -20.0]
$r->slack([[2, 1], [1, 1]], [100, 80]);   // [0.0, 0.0]: both resources used up
```

**Marginals (shadow prices)** are the change in the objective per unit
increase of each right-hand side, with SciPy's signs. Here one more hour of
assembly raises profit by 20, and one more hour of machining raises it by 10.
They tell you which constraint is worth relaxing. `eqlinMarginals` covers
equality rows and `reducedCosts` covers the variables.

### Bounds

| `bounds:` | Meaning |
|---|---|
| `null` (default) | every variable in `[0, +∞)` |
| `[lo, hi]` | the same bounds for every variable |
| `[[lo0, hi0], [lo1, hi1], …]` | one pair per variable |

Use `null` or `±INF` for an open side: `[null, null]` is a free variable.

### Status

| `status` | `success` | Meaning |
|---:|---|---|
| 0 | true | optimal |
| 1 | false | iteration limit (`maxiter`); `x` holds the last point |
| 2 | false | infeasible |
| 3 | false | unbounded |
| 4 | false | numerical difficulties |

Malformed input is rejected before the solver runs: wrong shapes throw
`ShapeError`, and NaN or ±INF in `c`, `A` or `b` (NaN in bounds) throws
`InvalidArgumentException` (`ValueError` from the extension), as SciPy does.
Infinite bounds are allowed.

## milp

```php
// Facility location: open any of 3 sites (fixed cost), serve 3 customers.
$fixed = [12, 10, 14];
$serve = [[4, 6, 9], [5, 4, 7], [6, 3, 4]];     // cost of serving customer j from site i
// variables: open_0..2, then assign_ij (row-major) — 12 binaries
$c = array_merge($fixed, ...$serve);
$A_eq = $b_eq = $A_ub = $b_ub = [];
for ($j = 0; $j < 3; $j++) {                    // each customer assigned once
    $row = array_fill(0, 12, 0);
    for ($i = 0; $i < 3; $i++) { $row[3 + 3 * $i + $j] = 1; }
    $A_eq[] = $row; $b_eq[] = 1;
}
for ($i = 0; $i < 3; $i++) {                    // only to open sites: assign_ij - open_i <= 0
    for ($j = 0; $j < 3; $j++) {
        $row = array_fill(0, 12, 0);
        $row[3 + 3 * $i + $j] = 1; $row[$i] = -1;
        $A_ub[] = $row; $b_ub[] = 0;
    }
}

$r = LP::milp($c, integrality: true, A_ub: $A_ub, b_ub: $b_ub,
              A_eq: $A_eq, b_eq: $b_eq, bounds: [0, 1]);
$r->x;          // open_1 = 1 only; all customers served from site 1
$r->fun;        // 26.0
$r->nodes;      // branch-and-bound nodes explored
$r->gap();      // relative gap between fun and bestBound (0 when proven optimal)
```

`integrality` is `true` (all variables integer), `false`, or a per-variable
list of 0/1 (SciPy's convention). `nodeLimit` caps the search; if it is hit
with a feasible solution in hand, `status` is 1 and `x` is the best solution
found so far. Check `gap()` to see how far from optimal it may be.
`mipRelGap` stops early once the gap is small enough.

## Method and limits

- **linprog:** dense two-phase primal simplex on a full tableau. Pricing is
  Dantzig's rule, switching to Bland's rule after a run of degenerate pivots
  (no cycling). The ratio test is Harris's two-pass test (stable pivots). Bounds
  are handled by substitution.
- **milp:** depth-first branch and bound on that simplex, branching on the
  most fractional variable and exploring the nearer child first.
- Checked against SciPy's HiGHS on 14 LPs (optimal, unbounded, infeasible,
  equality-constrained, free and bounded variables) and 10 MILPs: identical
  objective to 1e-9, identical marginals where the dual is unique.
- **Size:** the tableau is dense, `(m+1) × (n+1)` doubles. A 200 × 200 LP solves
  in about 10 ms. Beyond a few thousand rows or columns, memory and time grow
  quickly. Use a dedicated solver (HiGHS) for large sparse models; see the
  [roadmap](../project/roadmap.md).
- Branch and bound has no cutting planes or presolve. It suits problems with
  tens to a few hundred integer variables.

## Extension and Laravel

```php
Tessero\Ext\Engine::linprog($c, $A_ub, $b_ub, $A_eq, $b_eq, $bounds);   // array result
Tessero\Ext\Engine::milp($c, true, $A_ub, $b_ub, $A_eq, $b_eq, $bounds);

Tessero::linprog(...);   // Laravel facade: same keys on either backend
```

The array result has the keys `x`, `fun`, `success`, `status`, `message`,
`nit`, `ineqlin`, `eqlin` and `reduced_costs` (`nodes` and `best_bound` for
milp).
