# CsrMatrix

`Tessero\Sparse\CsrMatrix`

Compressed sparse row matrix of float64 (scipy.sparse.csr_matrix).
Column indices within a row are kept sorted and duplicate entries summed.

## Methods

### fromArrays

```php
static fromArrays(mixed $indptr, mixed $indices, mixed $data, array $shape): self
```

Wrap existing CSR arrays (validated).

### fromDense

```php
static fromDense(mixed $dense): self
```

### fromTriplets

```php
static fromTriplets(Tessero\NDArray|array $row, Tessero\NDArray|array $col, Tessero\NDArray|array $values, array $shape): self
```

From coordinate triplets (scipy.sparse.coo_matrix(...).tocsr()); duplicates are summed.

### identity

```php
static identity(int $n): self
```

### bicgstab

```php
bicgstab(mixed $b, mixed $x0 = null, float $rtol = 1.0E-5, float $atol = 0.0, ?int $maxiter = null, bool $throw = false): array
```

BiCGSTAB for general square systems (scipy.sparse.linalg.bicgstab).

### cg

```php
cg(mixed $b, mixed $x0 = null, float $rtol = 1.0E-5, float $atol = 0.0, ?int $maxiter = null, bool $throw = false): array
```

Conjugate gradients (scipy.sparse.linalg.cg) with Jacobi preconditioning; A must be SPD.

### data

```php
data(): Tessero\NDArray
```

### diagonal

```php
diagonal(): Tessero\NDArray
```

Main diagonal as a dense vector.

### dot

```php
dot(mixed $x): Tessero\NDArray
```

A @ x for a dense vector (n,) or matrix (n, k).

### indices

```php
indices(): Tessero\NDArray
```

### indptr

```php
indptr(): Tessero\NDArray
```

### nnz

```php
nnz(): int
```

### scale

```php
scale(float $alpha): self
```

### shape

```php
shape(): array
```

### toDense

```php
toDense(): Tessero\NDArray
```

### transpose

```php
transpose(): self
```
