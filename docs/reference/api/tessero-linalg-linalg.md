# Linalg

`Tessero\Linalg\Linalg`

numpy.linalg / scipy.linalg on LAPACK (row-major LAPACKE, float64).

Every routine works on a private contiguous float64 copy, so inputs are
never modified. Functions marked "stacked" accept (..., M, M) arrays and
loop over the leading dimensions like NumPy's gufuncs.

## Methods

### cholesky

```php
static cholesky(mixed $a, bool $upper = false): Tessero\NDArray
```

Lower-triangular Cholesky factor L with A = L L^T (stacked).

### cond

```php
static cond(mixed $a): float
```

### det

```php
static det(mixed $a): Tessero\NDArray|float
```

Determinant (stacked).

### eig

```php
static eig(mixed $a): array
```

General eigen-decomposition. Eigenvalues/vectors are float64 when all
eigenvalues are real, complex128 otherwise (as NumPy does).

### eigh

```php
static eigh(mixed $a, bool $vectors = true): array
```

Eigen-decomposition of a symmetric matrix (lower triangle is read), eigenvalues ascending (stacked).

### eigvals

```php
static eigvals(mixed $a): Tessero\NDArray
```

### eigvalsh

```php
static eigvalsh(mixed $a): Tessero\NDArray
```

### inv

```php
static inv(mixed $a): Tessero\NDArray
```

Inverse (stacked).

### lstsq

```php
static lstsq(mixed $a, mixed $b, ?float $rcond = null): array
```

Least squares (numpy.linalg.lstsq via LAPACK gelsd).

### lu

```php
static lu(mixed $a): array
```

LU factorisation with partial pivoting, A = P L U (scipy.linalg.lu).

### matrixRank

```php
static matrixRank(mixed $a, ?float $tol = null): int
```

### norm

```php
static norm(mixed $x, string|int|float|null $ord = null, ?int $axis = null): Tessero\NDArray|float
```

Vector or matrix norm (numpy.linalg.norm for ord null, 'fro', 'nuc', 1, 2, inf, -inf and vector p-norms).

### pinv

```php
static pinv(mixed $a, float $rcond = 1.0E-15): Tessero\NDArray
```

Moore-Penrose pseudo-inverse via SVD; singular values below rcond * max(s) are treated as zero.

### qr

```php
static qr(mixed $a, string $mode = 'reduced'): Tessero\NDArray|array
```

QR factorisation. mode 'reduced' (Q: M x K, R: K x N, K = min(M, N)),
'complete' (Q: M x M, R: M x N) or 'r' (R only).

### slogdet

```php
static slogdet(mixed $a): array
```

Sign and natural log of |det| (stacked), robust against overflow.

### solve

```php
static solve(mixed $a, mixed $b): Tessero\NDArray
```

Solve A x = b (stacked). b may be (M,) or (M, K).

### solveTriangular

```php
static solveTriangular(mixed $a, mixed $b, bool $lower = false, bool $transpose = false, bool $unitDiagonal = false): Tessero\NDArray
```

Solve a triangular system (scipy.linalg.solve_triangular).

### svd

```php
static svd(mixed $a, bool $fullMatrices = true, bool $computeUv = true): Tessero\NDArray|array
```

Singular value decomposition A = U diag(s) Vt.

### trace

```php
static trace(mixed $a): float
```

### tril

```php
static tril(Tessero\NDArray $a, int $k = 0): Tessero\NDArray
```

Lower triangle (numpy.tril).

### triu

```php
static triu(Tessero\NDArray $a, int $k = 0): Tessero\NDArray
```

Upper triangle (numpy.triu).
