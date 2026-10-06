# TesseroManager

`Tessero\Laravel\TesseroManager`

One entry point for applications, whichever backend is installed.

Arrays come back as Tessero\Ext\NDArray (native extension) or
Tessero\NDArray (FFI); both have the same core API (shape, dtype, slicing,
arithmetic, reductions, toArray, toJson, serialize). Solver results are
normalised to plain arrays so controllers and jobs do not care which
backend produced them.

## Methods

### __construct

```php
__construct(array $config = [])
```

### applyRuntimeSettings

```php
applyRuntimeSettings(): void
```

Push threads / budget / tolerance into the backend (called at boot and per Octane request).

### array

```php
array(mixed $data, ?string $dtype = null): Tessero\Ext\NDArray|Tessero\NDArray
```

### backend

```php
backend(): string
```

'ext' or 'ffi'

### config

```php
config(): array
```

### fromBytes

```php
fromBytes(string $bytes, string $dtype, array $shape): Tessero\Ext\NDArray|Tessero\NDArray
```

### info

```php
info(): array
```

### linprog

```php
linprog(mixed $c, mixed $A_ub = null, mixed $b_ub = null, mixed $A_eq = null, mixed $b_eq = null, ?array $bounds = null): array
```

scipy.optimize.linprog. Returns x, fun, success, status, message, nit, ineqlin, eqlin (marginals).

### load

```php
load(string $path, ?string $mmapMode = null): Tessero\Ext\NDArray|Tessero\NDArray
```

Read a .npy file (numpy.load); with $mmapMode ('r', 'r+', 'c') it is memory-mapped instead.

### memmap

```php
memmap(string $filename, string $mode = 'r+', ?array $shape = null, string $dtype = 'float64', int $offset = 0): Tessero\Ext\NDArray|Tessero\NDArray
```

A memory-mapped array: the file's pages are the array's memory, so files
larger than RAM can be processed. Modes as numpy.memmap: 'r', 'r+', 'w+',
'c'. Mapped bytes are not counted against memory_budget; info() reports
them as memory_mapped. Both backends share libtessero's mapping code.

### milp

```php
milp(mixed $c, array|bool $integrality = true, mixed $A_ub = null, mixed $b_ub = null, mixed $A_eq = null, mixed $b_eq = null, ?array $bounds = null): array
```

scipy.optimize.milp. Returns x, fun, success, status, message, nodes, best_bound.

### normal

```php
normal(int $seed, array|int $shape, float $mean = 0.0, float $sd = 1.0): Tessero\Ext\NDArray|Tessero\NDArray
```

NumPy-identical random numbers: numpy.random.default_rng($seed).normal(...).

### openMemmap

```php
openMemmap(string $path, string $mode = 'r+', ?array $shape = null, string $dtype = 'float64', bool $fortranOrder = false): Tessero\Ext\NDArray|Tessero\NDArray
```

A memory-mapped .npy file (numpy.lib.format.open_memmap); 'w+' creates it.

### random

```php
random(int $seed, array|int $shape): Tessero\Ext\NDArray|Tessero\NDArray
```

### solveMdp

```php
solveMdp(mixed $P, mixed $R, float $gamma, string $method = 'policy_iteration'): array
```

Solve a discounted MDP. $P: (actions, states, states), $R: (states, actions).
$method: 'policy_iteration' (exact) or 'value_iteration'.

### zeros

```php
zeros(array|int $shape, string $dtype = 'float64'): Tessero\Ext\NDArray|Tessero\NDArray
```
