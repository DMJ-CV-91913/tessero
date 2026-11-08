# Tessero

`Tessero\Tessero`

Process-wide settings and diagnostics.

## Constants

| Name | Value |
|---|---|
| `VERSION` | `'0.3.0'` |
| `ABI_VERSION` | `2` |

## Methods

### abiVersion

```php
static abiVersion(): int
```

The C ABI generation of the loaded libtessero.

### allocCacheBytes

```php
static allocCacheBytes(): int
```

Bytes of freed large blocks kept for reuse now.

### allocCacheLimit

```php
static allocCacheLimit(): int
```

### feature

```php
static feature(string $name): bool
```

Whether the loaded build has a capability: "repro_f32", "gemm_canonical", "lapack", "openblas_threads",
"openmp" or "fpenv_check". Unknown names give false.

### fftCacheBytes

```php
static fftCacheBytes(): int
```

Bytes the FFT plan cache holds now.

### fftCacheLimit

```php
static fftCacheLimit(): int
```

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

### setAllocCacheLimit

```php
static setAllocCacheLimit(int $bytes): void
```

Cap the large blocks this process keeps for reuse, in bytes (default 64 MiB, 16 MiB in a web SAPI, 0 = no
reuse). A freed array of 1 MiB or more is kept and handed out again for the next array of about its size, so a
repeated call does not page-fault its memory in again. Kept blocks are outside the memory budget and each
php-fpm worker keeps its own. Lowering the limit releases kept blocks at once.

### setBlasThreads

```php
static setBlasThreads(int $threads): void
```

Cap the BLAS thread pools: the BLAS this package loads over FFI (Linalg) and the one libtessero links
(matmul). Web SAPIs default to 1 (TESSERO_BLAS_THREADS overrides), since php-fpm runs a worker per core.

### setFftCacheLimit

```php
static setFftCacheLimit(int $bytes): void
```

Cap the FFT plan cache of this process in bytes (default 256 MiB, 0 = no caching). Repeated transforms of one
length reuse the cached plan; a prime length near 1M points needs about 112 MB of it. The cache is outside the
memory budget and each php-fpm worker holds its own, so lower it there. Lowering it frees idle plans at once.

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
