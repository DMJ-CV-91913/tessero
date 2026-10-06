# Engine

`Tessero\Ext\Engine` *(native extension)*

## Methods

### fft

```php
static fft($x, bool $inverse = …)
```

### getEpsilon

```php
static getEpsilon()
```

### getMaxThreads

```php
static getMaxThreads()
```

### info

```php
static info()
```

### integers

```php
static integers(int $seed, $shape, int $low, int $high)
```

### kernelVersion

```php
static kernelVersion()
```

### linprog

```php
static linprog($c, $A_ub = …, $b_ub = …, $A_eq = …, $b_eq = …, ?array $bounds = …, int $maxiter = …)
```

### mdpFiniteHorizon

```php
static mdpFiniteHorizon($P, $R, int $horizon, float $gamma = …)
```

### mdpPolicyIteration

```php
static mdpPolicyIteration($P, $R, float $gamma, int $evalSweeps = …, int $maxIter = …)
```

### mdpValueIteration

```php
static mdpValueIteration($P, $R, float $gamma, float $epsilon = …, int $maxIter = …)
```

### memoryBudget

```php
static memoryBudget()
```

### memoryInUse

```php
static memoryInUse()
```

### milp

```php
static milp($c, $integrality, $A_ub = …, $b_ub = …, $A_eq = …, $b_eq = …, ?array $bounds = …, int $nodeLimit = …, float $mipRelGap = …)
```

### normal

```php
static normal(int $seed, $shape, float $a = …, float $b = …)
```

### openmp

```php
static openmp()
```

### peakMemory

```php
static peakMemory()
```

### random

```php
static random(int $seed, $shape, float $a = …, float $b = …)
```

### setEpsilon

```php
static setEpsilon(float $value)
```

### setMaxThreads

```php
static setMaxThreads(int $value)
```

### setMemoryBudget

```php
static setMemoryBudget(int $value)
```

### simd

```php
static simd()
```

### version

```php
static version()
```
