# Tessero

`Tessero\Tessero`

Process-wide settings and diagnostics.

## Constants

| Name | Value |
|---|---|
| `VERSION` | `'0.2.0'` |
| `ABI_VERSION` | `2` |

## Methods

### abiVersion

```php
static abiVersion(): int
```

The C ABI generation of the loaded libtessero.

### feature

```php
static feature(string $name): bool
```

Whether the loaded build has a capability: "repro_f32", "gemm_canonical", "lapack", "openblas_threads",
"openmp" or "fpenv_check". Unknown names give false.

### info

```php
static info(): array
```

### memoryBudget

```php
static memoryBudget(): int
```

### memoryInUse

```php
static memoryInUse(): int
```

### peakMemory

```php
static peakMemory(): int
```

### rng

```php
static rng(array|int|null $seed = null): Tessero\Random\Generator
```

### selfcheck

```php
static selfcheck(): array
```

Numerical self-check of the loaded libtessero (docs/reference/numerical-guarantees.md): the FP control state
of this thread and the OpenMP workers, then the known-answer tests. fpenv is 0 when the FP environment is
canonical (round to nearest, no flush-to-zero, traps masked), else a bitmask named in fpenv_reason;
selfcheck is 0 when every probe matches, a negative code otherwise (tessero.h).

### setBlasThreads

```php
static setBlasThreads(int $threads): void
```

Cap the BLAS thread pools: the BLAS this package loads over FFI (Linalg) and the one libtessero links
(matmul). Web SAPIs default to 1 (TESSERO_BLAS_THREADS overrides), since php-fpm runs a worker per core.

### setMemoryBudget

```php
static setMemoryBudget(int $bytes): void
```

Cap native memory for this process (0 = unlimited). Allocations over the
cap throw MemoryError instead of letting a web worker grow without bound;
native buffers do not count toward memory_limit.

### setThreads

```php
static setThreads(int $threads): void
```

Threads for large element-wise kernels and MDP sweeps (OpenMP builds).
Default 1: PHP-FPM already runs a worker per core. Results are identical
for any thread count.

### threads

```php
static threads(): int
```
