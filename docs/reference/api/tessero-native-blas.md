# Blas

`Tessero\Native\Blas`

Optional BLAS + LAPACKE binding (OpenBLAS, LP64 / 32-bit integers).

Tried in order: $TESSERO_BLAS, the system OpenBLAS, and the OpenBLAS that
SciPy's wheels bundle (symbols prefixed "scipy_"; `pip install scipy` is
the no-root way to get a tuned BLAS). ILP64 builds (libopenblas64_,
numpy's bundled copy) are refused: their integers are 64-bit and passing
32-bit ints would corrupt memory.

Without a BLAS, matmul falls back to libtessero's blocked kernel and
Linalg throws LibraryUnavailable.

## Constants

| Name | Value |
|---|---|
| `ROW_MAJOR` | `101` |
| `NO_TRANS` | `111` |
| `TRANS` | `112` |
| `INT_MAX` | `2147483647` |

## Methods

### available

```php
static available(): bool
```

### call

```php
static call(string $fn, mixed ...$args): mixed
```

Call a BLAS/LAPACKE symbol by its unprefixed name. Integer arguments must fit the library's 32-bit
int: PHP FFI silently truncates a wider value (2^32 + 3 becomes 3), which would make LAPACK work on
a different size than the buffers have.

### ffi

```php
static ffi(): FFI
```

### gemm

```php
static gemm(Tessero\DType $dt, int $m, int $n, int $k, FFI\CData $a, FFI\CData $b, FFI\CData $c): void
```

C = A (m x k) @ B (k x n), all row-major and contiguous.

### library

```php
static library(): ?string
```

### preloadCandidates

```php
static preloadCandidates(): array
```

[path, prefix, header] triples for resources/preload.php.

### ptr

```php
static ptr(string $type, int $address): FFI\CData
```

Pointer of the given C type in the BLAS FFI scope at an absolute address. @internal

### setThreads

```php
static setThreads(int $n): void
```
