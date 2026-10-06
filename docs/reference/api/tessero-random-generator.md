# Generator

`Tessero\Random\Generator`

PCG64 generator seeded through NumPy's SeedSequence.

  Generator::defaultRng(42)->random(5)   produces exactly   numpy.random.default_rng(42).random(5)

Stream-identical to NumPy: every distribution method (GeneratorMethods, generated from the kernel's
function registry and drawn with NumPy's own distribution code), integers, choice (with replacement,
with or without p), permutation and shuffle.

## Methods

### defaultRng

```php
static defaultRng(array|int|null $seed = null): self
```

### beta

```php
beta(mixed $a, mixed $b, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a beta distribution (Generator.beta).

numpy.random.Generator.beta

### binomial

```php
binomial(mixed $n, mixed $p, array|int|null $size = null): Tessero\NDArray|int
```

Samples from a binomial distribution (Generator.binomial).

numpy.random.Generator.binomial

### chisquare

```php
chisquare(mixed $df, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a chi-square distribution (Generator.chisquare).

numpy.random.Generator.chisquare

### choice

```php
choice(Tessero\NDArray|array|int $a, array|int|null $size = null, bool $replace = true, ?array $p = null): mixed
```

Sample from a 1-D array (or arange(a) for an int). Probabilities $p use
NumPy's CDF-inversion, so the draws match numpy's choice(..., p=p).

### exponential

```php
exponential(mixed $scale = 1.0, array|int|null $size = null): Tessero\NDArray|float
```

Samples from an exponential distribution (Generator.exponential).

numpy.random.Generator.exponential

### f

```php
f(mixed $dfnum, mixed $dfden, array|int|null $size = null): Tessero\NDArray|float
```

Samples from an F distribution (Generator.f).

numpy.random.Generator.f

### gamma

```php
gamma(mixed $shape, mixed $scale = 1.0, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a gamma distribution (Generator.gamma).

numpy.random.Generator.gamma

### geometric

```php
geometric(mixed $p, array|int|null $size = null): Tessero\NDArray|int
```

Samples from the geometric distribution (Generator.geometric).

numpy.random.Generator.geometric

### getState

```php
getState(): array
```

### gumbel

```php
gumbel(mixed $loc = 0.0, mixed $scale = 1.0, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a Gumbel distribution (Generator.gumbel).

numpy.random.Generator.gumbel

### hypergeometric

```php
hypergeometric(mixed $ngood, mixed $nbad, mixed $nsample, array|int|null $size = null): Tessero\NDArray|int
```

Samples from a hypergeometric distribution (Generator.hypergeometric).

numpy.random.Generator.hypergeometric

### integers

```php
integers(int $low, ?int $high = null, array|int|null $size = null): Tessero\NDArray|int
```

Integers in [low, high); with one bound, [0, low).

### laplace

```php
laplace(mixed $loc = 0.0, mixed $scale = 1.0, array|int|null $size = null): Tessero\NDArray|float
```

Samples from the Laplace or double exponential distribution (Generator.laplace).

numpy.random.Generator.laplace

### logistic

```php
logistic(mixed $loc = 0.0, mixed $scale = 1.0, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a logistic distribution (Generator.logistic).

numpy.random.Generator.logistic

### lognormal

```php
lognormal(mixed $mean = 0.0, mixed $sigma = 1.0, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a log-normal distribution (Generator.lognormal).

numpy.random.Generator.lognormal

### logseries

```php
logseries(mixed $p, array|int|null $size = null): Tessero\NDArray|int
```

Samples from a logarithmic series distribution (Generator.logseries).

numpy.random.Generator.logseries

### negativeBinomial

```php
negativeBinomial(mixed $n, mixed $p, array|int|null $size = null): Tessero\NDArray|int
```

Samples from a negative binomial distribution (Generator.negative_binomial).

numpy.random.Generator.negative_binomial

### noncentralChisquare

```php
noncentralChisquare(mixed $df, mixed $nonc, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a noncentral chi-square distribution (Generator.noncentral_chisquare).

numpy.random.Generator.noncentral_chisquare

### noncentralF

```php
noncentralF(mixed $dfnum, mixed $dfden, mixed $nonc, array|int|null $size = null): Tessero\NDArray|float
```

Samples from the noncentral F distribution (Generator.noncentral_f).

numpy.random.Generator.noncentral_f

### normal

```php
normal(mixed $loc = 0.0, mixed $scale = 1.0, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a normal (Gaussian) distribution (Generator.normal).

numpy.random.Generator.normal

### pareto

```php
pareto(mixed $a, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a Pareto II (Lomax) distribution (Generator.pareto).

numpy.random.Generator.pareto

### permutation

```php
permutation(Tessero\NDArray|array|int $x): Tessero\NDArray
```

Random permutation of arange(n), or a shuffled copy of an array along axis 0.

### poisson

```php
poisson(mixed $lam = 1.0, array|int|null $size = null): Tessero\NDArray|int
```

Samples from a Poisson distribution (Generator.poisson).

numpy.random.Generator.poisson

### power

```php
power(mixed $a, array|int|null $size = null): Tessero\NDArray|float
```

Samples in [0, 1] from a power distribution with positive exponent a - 1 (Generator.power).

numpy.random.Generator.power

### random

```php
random(array|int|null $size = null, string $dtype = 'float64'): Tessero\NDArray|float
```

Floats in the half-open interval [0.0, 1.0) (Generator.random).

numpy.random.Generator.random

### rayleigh

```php
rayleigh(mixed $scale = 1.0, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a Rayleigh distribution (Generator.rayleigh).

numpy.random.Generator.rayleigh

### setState

```php
setState(array $words): void
```

### shuffle

```php
shuffle(Tessero\NDArray $x): void
```

Shuffle along axis 0 in place.

### standardCauchy

```php
standardCauchy(array|int|null $size = null): Tessero\NDArray|float
```

Samples from a standard Cauchy distribution (Generator.standard_cauchy).

numpy.random.Generator.standard_cauchy

### standardExponential

```php
standardExponential(array|int|null $size = null, string $dtype = 'float64', string $method = 'zig'): Tessero\NDArray|float
```

Samples from the standard exponential distribution (Generator.standard_exponential).

numpy.random.Generator.standard_exponential

### standardGamma

```php
standardGamma(mixed $shape, array|int|null $size = null, string $dtype = 'float64'): Tessero\NDArray|float
```

Samples from the standard gamma distribution (Generator.standard_gamma).

numpy.random.Generator.standard_gamma

### standardNormal

```php
standardNormal(array|int|null $size = null, string $dtype = 'float64'): Tessero\NDArray|float
```

Samples from the standard normal distribution, ziggurat method (Generator.standard_normal).

numpy.random.Generator.standard_normal

### standardT

```php
standardT(mixed $df, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a standard Student's t distribution (Generator.standard_t).

numpy.random.Generator.standard_t

### stateHandle

```php
stateHandle(): FFI\CData
```

The PCG64 state vector, for the kernel calls that draw from this generator (scipy.stats rvs).

### triangular

```php
triangular(mixed $left, mixed $mode, mixed $right, array|int|null $size = null): Tessero\NDArray|float
```

Samples from the triangular distribution over [left, right] (Generator.triangular).

numpy.random.Generator.triangular

### uniform

```php
uniform(mixed $low = 0.0, mixed $high = 1.0, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a uniform distribution over [low, high) (Generator.uniform).

numpy.random.Generator.uniform

### vonmises

```php
vonmises(mixed $mu, mixed $kappa, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a von Mises distribution (Generator.vonmises).

numpy.random.Generator.vonmises

### wald

```php
wald(mixed $mean, mixed $scale, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a Wald, or inverse Gaussian, distribution (Generator.wald).

numpy.random.Generator.wald

### weibull

```php
weibull(mixed $a, array|int|null $size = null): Tessero\NDArray|float
```

Samples from a Weibull distribution (Generator.weibull).

numpy.random.Generator.weibull

### zipf

```php
zipf(mixed $a, array|int|null $size = null): Tessero\NDArray|int
```

Samples from a Zipf distribution (Generator.zipf).

numpy.random.Generator.zipf
