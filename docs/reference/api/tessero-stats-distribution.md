# Distribution

`Tessero\Stats\Distribution`

A frozen scipy.stats distribution: Stats::gamma(2.5, loc: 1.0, scale: 3.0).

Every method is one element-wise call into the kernel's distribution machinery (ADR 0011): SciPy's
rv_continuous / rv_discrete ported with the same support handling and generic fallbacks, so the results
match scipy.stats. Parameters may be numbers or arrays; they broadcast against x as in SciPy. The native
extension's Tessero\Ext\Distribution has the same methods and results.

## Methods

### __construct

```php
__construct(string $name, array $params)
```

### args

```php
args(): array
```

### cdf

```php
cdf(mixed $x): Tessero\NDArray|float
```

### entropy

```php
entropy(): Tessero\NDArray|float
```

### interval

```php
interval(float $confidence): array
```

### isf

```php
isf(mixed $q): Tessero\NDArray|float
```

### logcdf

```php
logcdf(mixed $x): Tessero\NDArray|float
```

### logpdf

```php
logpdf(mixed $x): Tessero\NDArray|float
```

### logpmf

```php
logpmf(mixed $k): Tessero\NDArray|float
```

### logsf

```php
logsf(mixed $x): Tessero\NDArray|float
```

### mean

```php
mean(): Tessero\NDArray|float
```

### median

```php
median(): Tessero\NDArray|float
```

### moment

```php
moment(int $order): Tessero\NDArray|float
```

Non-central moment of the given order.

### name

```php
name(): string
```

The SciPy name ("gamma").

### pdf

```php
pdf(mixed $x): Tessero\NDArray|float
```

### pmf

```php
pmf(mixed $k): Tessero\NDArray|float
```

Probability mass function (discrete distributions).

### ppf

```php
ppf(mixed $q): Tessero\NDArray|float
```

### rvs

```php
rvs(array|int|null $size = null, ?Tessero\Random\Generator $randomState = null): Tessero\NDArray|int|float
```

Random variates, drawn from $randomState's stream exactly as scipy.stats draws them from a
numpy.random.Generator (rvs(size, random_state=numpy.random.default_rng(seed))).

### sf

```php
sf(mixed $x): Tessero\NDArray|float
```

### stats

```php
stats(string $moments = 'mv'): Tessero\NDArray|array|float
```

Mean ('m'), variance ('v'), skew ('s') and/or excess kurtosis ('k').

                                                   mean, var, skew, kurtosis (in that order)

### std

```php
std(): Tessero\NDArray|float
```

### support

```php
support(): array
```

### var

```php
var(): Tessero\NDArray|float
```
