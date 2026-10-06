# Sparse matrices

`Tessero\Sparse\CsrMatrix` is a float64 compressed-sparse-row matrix, the
same layout as `scipy.sparse.csr_matrix`.

```php
use Tessero\Sparse\CsrMatrix;

// tridiagonal, diagonally dominant: 4 on the diagonal, -1 either side
$n = 100_000;
$rows = []; $cols = []; $vals = [];
for ($i = 0; $i < $n; $i++) {
    $rows[] = $i; $cols[] = $i; $vals[] = 4.0;
    if ($i > 0)      { $rows[] = $i; $cols[] = $i - 1; $vals[] = -1.0; }
    if ($i < $n - 1) { $rows[] = $i; $cols[] = $i + 1; $vals[] = -1.0; }
}
$A = CsrMatrix::fromTriplets($rows, $cols, $vals, [$n, $n]);   // duplicates are summed

$A->nnz();                     // 299 998
$A->dot($x);                   // sparse × dense vector or matrix
$A->transpose();
$A->diagonal();
$A->scale(0.5);
$A->toDense();                 // careful with large n

[$x, $info] = $A->cg($b, rtol: 1e-10);          // symmetric positive definite
[$x, $info] = $A->bicgstab($b, rtol: 1e-10);    // general square
```

Other constructors: `fromDense($array)`,
`fromArrays($indptr, $indices, $data, $shape)` and `identity($n)`.

## Iterative solvers

`cg` (conjugate gradient) and `bicgstab` use a Jacobi (diagonal)
preconditioner and SciPy's stopping rule, `‖r‖ ≤ max(rtol·‖b‖, atol)`. They
return `[$x, $info]`. `info` is 0 on convergence and otherwise the number of
iterations run. With `throw: true` they throw `ConvergenceError` instead.

The whole iteration runs in C: one PHP call, no per-iteration overhead. The
example above (100 000 unknowns) converges to rtol 1e-10 in about 20 ms
(measured on a 2-core VM). Iteration counts grow with the condition number: the 1-D
Poisson matrix (2 on the diagonal) of the same size needs tens of thousands of
CG iterations, whatever the implementation.

## Limits

There is no direct sparse solver (`spsolve`), no CSC/COO classes and no
sparse × sparse product yet ([roadmap](../project/roadmap.md)). For
Markov decision processes the transition matrix is stored as CSR internally;
see [MDP](mdp.md).
