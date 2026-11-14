# LeastSquares

`Tessero\Optimize\LeastSquares`

Non-linear least squares by Levenberg-Marquardt: solve / leastSquares with Nielsen's damping update and
forward-difference Jacobians, curveFit with MINPACK's lmdif as scipy.optimize.curve_fit runs it.

## Methods

### curveFit

```php
static curveFit(callable $f, object|array $xdata, object|array $ydata, array $p0, object|array|null $sigma = null, bool $absoluteSigma = false, bool $fullOutput = false): array
```

scipy.optimize.curve_fit (method 'lm', SciPy's choice without bounds): fit f(x, ...params) to data with
MINPACK's lmdif as SciPy runs it (linalg.lmdif: SciPy's own code, leastsq's defaults), so the parameters, the
number of evaluations and the covariance follow SciPy's. Returns [popt, pcov], with $fullOutput
[popt, pcov, infodict (nfev, fvec), mesg, ier] as SciPy's full_output.

The model is called the way xdata comes in. A PHP list: once per point, f(float $x, float ...$p) returning a
number. An NDArray (of either backend): on all of xdata, f(NDArray $x, float ...$p) returning one value per
point (an NDArray of that backend or a list), as SciPy calls it, the data staying in the kernel. For the n
evaluations of each forward-difference Jacobian such a model is first offered them in one call, x as a column
(m, 1) and each parameter as a row (1, n) of the n evaluation points; it keeps that call when it returns an
(m, n) array with the same bits as the n separate calls (compared once), and is called n times otherwise.

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
