# Tessero

`Tessero\Tessero`

Process-wide settings and diagnostics.

## Constants

| Name | Value |
|---|---|
| `VERSION` | `'0.2.0'` |

## Methods

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

### setBlasThreads

```php
static setBlasThreads(int $threads): void
```

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
