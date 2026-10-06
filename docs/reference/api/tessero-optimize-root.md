# Root

`Tessero\Optimize\Root`

Scalar and vector root finding (scipy.optimize.brentq, newton, root).

## Methods

### brentq

```php
static brentq(callable $f, float $a, float $b, float $xtol = 2.0E-12, ?float $rtol = null, int $maxiter = 100, bool $fullOutput = false): array|float
```

Brent's method on a bracketing interval - a line-by-line port of
scipy/optimize/Zeros/brentq.c, so it returns the same root in the same
number of evaluations.

### fdJacobian

```php
static fdJacobian(callable $F, array $x, array $fx): array
```

### newton

```php
static newton(callable $f, float $x0, ?callable $fprime = null, float $tol = 1.48E-8, int $maxiter = 50, float $rtol = 0.0): float
```

Newton-Raphson (with fprime) or secant (without), as scipy.optimize.newton for scalars.

### root

```php
static root(callable $fun, array $x0, ?callable $jac = null, float $tol = 1.49012E-8, int $maxiter = 100): Tessero\Optimize\OptimizeResult
```

Solve F(x) = 0 for a vector function with damped Newton steps
(Jacobian by forward differences unless given; linear solves by LAPACK
when available, partial-pivot Gaussian elimination otherwise).
