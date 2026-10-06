# Registry

`Tessero\Native\Registry`

Calls into libtessero's function registry (ADR 0011): scipy.special functions, statistics and the other
SciPy-style routines. The kernel describes every function (arguments, outputs, parameters) and this class
interprets that description, so each registry entry is callable without a hand-written binding; the
native extension does the same in C (ext/tessero_fn.c), which keeps the two backends identical.

Façade classes (Tessero\Special, Tessero\Stats, ...) are generated from the registry by
tools/parity/gen-facades.php and forward here.

## Methods

### all

```php
static all(): array
```

### defaultValue

```php
static defaultValue(string $d): mixed
```

A registry default ("None", "True", "10", "'xy'") as a PHP value.

### gufunc

```php
static gufunc(string $name, array $inputs, array|int|null $axis, bool $keepdims, array $params): Tessero\NDArray|array|int|float|bool
```

Evaluate a generalised ufunc (reductions and other functions over an axis).

### info

```php
static info(string $name): array
```

### label

```php
static label(string $name): string
```

### params

```php
static params(array $info, array $given, string $label): array
```

Parameter values in registry order: numbers as given, booleans 0/1, null NAN, enum names by index.

### random

```php
static random(string $name, FFI\CData $state, array $inputs, array|int|null $size, array $params = []): Tessero\NDArray|int|float
```

Draw variates with a Generator method of the registry (kind "random"), NumPy's stream exactly.

### routine

```php
static routine(string $name, array $args): mixed
```

Call a routine (functions whose output sizes depend on the data: unique, histogram, cov, ...).
Arguments go to the kernel as numbers, strings, arrays or null; array results are adopted without a copy.

### routineVariadic

```php
static routineVariadic(string $name, array $given): mixed
```

A routine with a variadic argument (numpy's `*xi`): the positional arguments form that sequence, named
arguments (collected by the façade's `...$args`) fill the keyword-only arguments after it.

### ufunc

```php
static ufunc(string $name, array $args, Tessero\NDArray|array|null $out = null, int $method = -1): Tessero\NDArray|array|float
```

Evaluate an element-wise registry function (a ufunc, or a distribution method when $method >= 0 on a
"dist" entry). All-scalar inputs give floats; otherwise arrays broadcast as in NumPy. Several outputs
come back as an array keyed by output name.
