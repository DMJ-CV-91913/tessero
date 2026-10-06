# Generator

`Tessero\Ext\Random\Generator` *(native extension)*

## Methods

### defaultRng

```php
static defaultRng($seed = null)
```

### beta

```php
beta($a, $b, $size = null)
```

### binomial

```php
binomial($n, $p, $size = null)
```

### chisquare

```php
chisquare($df, $size = null)
```

### exponential

```php
exponential($scale = 1.0, $size = null)
```

### f

```php
f($dfnum, $dfden, $size = null)
```

### gamma

```php
gamma($shape, $scale = 1.0, $size = null)
```

### geometric

```php
geometric($p, $size = null)
```

### getState

```php
getState()
```

### gumbel

```php
gumbel($loc = 0.0, $scale = 1.0, $size = null)
```

### hypergeometric

```php
hypergeometric($ngood, $nbad, $nsample, $size = null)
```

### integers

```php
integers($low, $high = null, $size = null, $endpoint = false)
```

### laplace

```php
laplace($loc = 0.0, $scale = 1.0, $size = null)
```

### logistic

```php
logistic($loc = 0.0, $scale = 1.0, $size = null)
```

### lognormal

```php
lognormal($mean = 0.0, $sigma = 1.0, $size = null)
```

### logseries

```php
logseries($p, $size = null)
```

### negativeBinomial

```php
negativeBinomial($n, $p, $size = null)
```

### noncentralChisquare

```php
noncentralChisquare($df, $nonc, $size = null)
```

### noncentralF

```php
noncentralF($dfnum, $dfden, $nonc, $size = null)
```

### normal

```php
normal($loc = 0.0, $scale = 1.0, $size = null)
```

### pareto

```php
pareto($a, $size = null)
```

### permutation

```php
permutation($x)
```

### poisson

```php
poisson($lam = 1.0, $size = null)
```

### power

```php
power($a, $size = null)
```

### random

```php
random($size = null, $dtype = 'float64')
```

### rayleigh

```php
rayleigh($scale = 1.0, $size = null)
```

### setState

```php
setState($words)
```

### shuffle

```php
shuffle($x)
```

### standardCauchy

```php
standardCauchy($size = null)
```

### standardExponential

```php
standardExponential($size = null, $dtype = 'float64', $method = 'zig')
```

### standardGamma

```php
standardGamma($shape, $size = null, $dtype = 'float64')
```

### standardNormal

```php
standardNormal($size = null, $dtype = 'float64')
```

### standardT

```php
standardT($df, $size = null)
```

### triangular

```php
triangular($left, $mode, $right, $size = null)
```

### uniform

```php
uniform($low = 0.0, $high = 1.0, $size = null)
```

### vonmises

```php
vonmises($mu, $kappa, $size = null)
```

### wald

```php
wald($mean, $scale, $size = null)
```

### weibull

```php
weibull($a, $size = null)
```

### zipf

```php
zipf($a, $size = null)
```
