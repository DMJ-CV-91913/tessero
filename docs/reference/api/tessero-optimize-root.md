# Root

`Tessero\Optimize\Root`

Scalar and vector root finding (scipy.optimize.brentq, newton, root).

## Methods

### bisect

```php
static bisect(callable $f, float $a, float $b, float $xtol = 2.0E-12, ?float $rtol = null, int $maxiter = 100): float
```

Bisection on a bracketing interval - a port of scipy/optimize/Zeros/bisect.c, returning the same root.

### brenth

```php
static brenth(callable $f, float $a, float $b, float $xtol = 2.0E-12, ?float $rtol = null, int $maxiter = 100): float
```

Brent's method with hyperbolic extrapolation - a port of scipy/optimize/Zeros/brenth.c (identical to brentq
except for the extrapolation step), returning the same root.

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

### fixedPoint

```php
static fixedPoint(callable $func, float $x0, float $xtol = 1.0E-8, int $maxiter = 500): float
```

scipy.optimize.fixed_point: a scalar fixed point of func (func(x) = x) by Steffensen's method with Aitken's
del^2 acceleration, as scipy's default method='del2'. Matches scipy iterate for iterate.

### fsolve

```php
static fsolve(callable $func, array $x0): array
```

scipy.optimize.fsolve: find a root of a vector function, returning the solution vector (the default
full_output=0 behaviour). A thin wrapper over root(); for a well-posed system the root is unique, so it
matches scipy (whose fsolve uses MINPACK hybr).

### newton

```php
static newton(callable $f, float $x0, ?callable $fprime = null, float $tol = 1.48E-8, int $maxiter = 50, float $rtol = 0.0): float
```

Newton-Raphson (with fprime) or secant (without), as scipy.optimize.newton for scalars.

### ridder

```php
static ridder(callable $f, float $a, float $b, float $xtol = 2.0E-12, ?float $rtol = null, int $maxiter = 100): float
```

Ridder's method on a bracketing interval - a port of scipy/optimize/Zeros/ridder.c, returning the same root.

### root

```php
static root(callable $fun, array $x0, ?callable $jac = null, float $tol = 1.49012E-8, int $maxiter = 100): Tessero\Optimize\OptimizeResult
```

### rootScalar

```php
static rootScalar(callable $f, ?string $method = null, ?array $bracket = null, ?float $x0 = null, ?float $x1 = null): array
```

scipy.optimize.root_scalar: find a root of a scalar function by dispatching to a named solver. Bracketing
methods ('brentq', 'bisect', 'brenth', 'ridder') need a two-element bracket; derivative-free point methods
('newton', 'secant') need a starting point x0. When method is null it is selected as scipy does: 'brentq'
if a bracket is given, else 'secant' if x1 is given, else 'newton'. Returns the result as
['root' => float, 'converged' => bool, 'flag' => string].

### toms748

```php
static toms748(callable $f, float $a, float $b, float $xtol = 2.0E-12, ?float $rtol = null, int $maxiter = 100): float
```

Algorithm 748 of Alefeld, Potra and Shi (the k=1 form: a Newton-quadratic interpolation step, a
double-length secant step and a bisection safeguard), as scipy.optimize.toms748. Converges to the same
root as scipy.
