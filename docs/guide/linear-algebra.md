# Linear algebra

`Tessero\Linalg\Linalg` covers `numpy.linalg` and the most used parts of
`scipy.linalg`. It calls LAPACK through LAPACKE (row-major) from an LP64
OpenBLAS. It is part of the FFI package; the extension does not include it.

```php
use Tessero\Linalg\Linalg;

$A = arr([[4.0, 1.0], [1.0, 3.0]]);
$b = arr([1.0, 2.0]);

Linalg::solve($A, $b);            // [0.0909..., 0.6363...]   (dgesv)
Linalg::inv($A);
Linalg::det($A);                  // 11.000000000000002 (LU in floating point, like NumPy)
[$sign, $logdet] = Linalg::slogdet($A);
Linalg::cholesky($A);             // lower triangular; upper: true for U
[$w, $V] = Linalg::eigh($A);      // symmetric: ascending eigenvalues
[$w, $V] = Linalg::eig($A);       // general: complex128 when any eigenvalue is complex (as NumPy)
[$U, $s, $Vt] = Linalg::svd($A);
[$Q, $R] = Linalg::qr($A);        // 'reduced' (default), 'complete', 'r'
[$P, $L, $U] = Linalg::lu($A);    // scipy.linalg.lu convention: A = P L U
Linalg::pinv($A);
Linalg::lstsq($X, $y);            // [solution, residuals, rank, singular values] (dgelsd)
Linalg::solveTriangular($L, $b, lower: true);
Linalg::norm($A);                 // Frobenius; ord: 1, 2, INF, -INF, 'fro', 'nuc'
Linalg::cond($A); Linalg::matrixRank($A); Linalg::trace($A);
```

## Stacks of matrices

`solve`, `inv`, `det`, `cholesky` and `eigh` accept arrays of shape
`[..., n, n]` and work on each matrix, as NumPy does. A batch of 10 000 2×2
systems is one PHP call.

## Errors

| Exception | When |
|---|---|
| `SingularMatrix` | `solve`, `inv` on an exactly singular matrix (LAPACK `info > 0`) |
| `NotPositiveDefinite` | `cholesky` |
| `ConvergenceError` | `eig`, `eigh`, `svd` did not converge |
| `ShapeError` | not square, or incompatible right-hand side |
| `LibraryUnavailable` | no LP64 OpenBLAS with LAPACKE was found |

All extend `Tessero\Exceptions\LinAlgError` (itself a `TesseroException`).

## Accuracy

Every function is checked against `numpy.linalg`/`scipy.linalg` on fixed
fixtures to a relative tolerance of 1e-9; most agree to about 1e-15.
Eigenvectors and singular vectors are compared up to sign, because LAPACK
builds may choose different signs.

## Threads

OpenBLAS starts one thread per core by default. Under PHP-FPM that
oversubscribes the machine (every worker would start N threads), so Tessero
sets OpenBLAS to 1 thread in non-CLI SAPIs. Change it with
`Tessero::setBlasThreads(n)` or `TESSERO_BLAS_THREADS`.
