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

## SciPy-style solvers on sparse input

The `scipy.sparse.linalg` functions in `Tessero\SparseLinalg` (and
`Tessero\Ext\SparseLinalg`) take a sparse matrix as well as a dense one:
`spsolve`, `eigsh` and the iterative solvers `cg`, `bicgstab`, `bicg`, `cgs`,
`minres`, `gmres`, `lgmres`, `gcrotmk`, `qmr` and `tfqmr`. A sparse matrix is
the array that `Sparse::diags`, `Sparse::csrArray` and the other sparse
routines return (keys `data`, `indices`, `indptr`, `shape`; both backends)
or a `CsrMatrix` (FFI backend). Its structure is checked before use, so a
malformed one is refused with an error.

```php
use Tessero\Sparse;
use Tessero\SparseLinalg;

$n = 100_000;
$A = Sparse::diags([array_fill(0, $n - 1, -1.0), array_fill(0, $n, 4.0), array_fill(0, $n - 1, -1.0)], [-1, 0, 1], [$n, $n]);
$x = SparseLinalg::spsolve($A, $b);      // banded LU: about 15 ms
$x = SparseLinalg::cg($A, $b);
$w = SparseLinalg::eigsh($A, 6);         // the 6 eigenvalues of largest magnitude
```

- **Iterative solvers** run the same iteration as on a dense matrix, with
  sparse products: the same result (but for the sign of an exact zero), at a
  cost of the stored entries instead of n² per product.
- **`spsolve`** factors the matrix's band (LAPACK `dgbsv`, partial pivoting),
  the bandwidths measured from the stored entries: tridiagonal and grid
  matrices solve at any size. A matrix whose band is too wide is solved dense
  when that fits in memory (2 GiB), and is refused otherwise.
- **`eigsh`** runs a thick-restart Lanczos iteration with full
  reorthogonalization, then searches the complement of the converged vectors so
  that a repeated eigenvalue is not missed. `which` = `LM`, `LA`, `SA`, `BE`;
  `SM` is solved dense up to n = 4096 (Lanczos reaches the smallest magnitudes
  slowly without a shift). `v0`, `ncv`, `maxiter` and `tol` apply; `sigma`
  (shift-invert) is not supported.

## Limits

There is no general sparse LU (`spsolve` is banded, see above), no CSC/COO
classes and no sparse × sparse product yet ([roadmap](../project/roadmap.md)).
For Markov decision processes the transition matrix is stored as CSR
internally; see [MDP](mdp.md).
