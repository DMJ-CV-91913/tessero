# LeastSquares

`Tessero\Optimize\LeastSquares`

Non-linear least squares by Levenberg-Marquardt (scipy.optimize.least_squares
method='lm' / curve_fit) with Nielsen's damping update and forward-difference
Jacobians.

## Methods

### curveFit

```php
static curveFit(callable $f, array $xdata, array $ydata, array $p0, ?array $sigma = null, bool $absoluteSigma = false): array
```

scipy.optimize.curve_fit: fit f(x, ...params) to data. Returns [popt, pcov].

### isotonicRegression

```php
static isotonicRegression(array $y, ?array $weights = null, bool $increasing = true): array
```

Isotonic (monotonic) regression by the Pool Adjacent Violators Algorithm
(scipy.optimize.isotonic_regression): the weighted least-squares best monotonic fit to $y.

### leastSquares

```php
static leastSquares(callable $fun, array $x0): Tessero\Optimize\OptimizeResult
```

scipy.optimize.least_squares (unbounded): minimise 0.5 ||fun(x)||^2 from x0. For an unbounded problem the
minimiser is that of the Levenberg-Marquardt solve (self::solve), so this wraps it in an OptimizeResult.
Bounds / the trf/dogbox trust-region variants are not yet supported.

### lsqLinear

```php
static lsqLinear(array $A, array $b, ?array $bounds = null): Tessero\Optimize\OptimizeResult
```

scipy.optimize.lsq_linear (unbounded): the linear least-squares solution of min 0.5||A x - b||^2, returned
as an OptimizeResult. The unbounded solution is the ordinary least-squares solution (normal equations);
finite bounds (the trf/bvls variants) are not yet supported.

### nnls

```php
static nnls(array $A, array $b, ?int $maxiter = null): array
```

### solve

```php
static solve(callable $residuals, array $p0, ?callable $jac = null, float $ftol = 1.0E-8, float $xtol = 1.0E-8, float $gtol = 1.0E-8, ?int $maxfev = null): array
```

Minimise 0.5 * sum(residuals(p)^2).
