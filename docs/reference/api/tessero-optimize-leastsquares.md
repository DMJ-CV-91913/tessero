# LeastSquares

`Tessero\Optimize\LeastSquares`

Non-linear least squares with MINPACK's Levenberg-Marquardt code as SciPy runs it (kernel linalg.lmdif): solve as
scipy.optimize.leastsq, curveFit as curve_fit, leastSquares as least_squares with method 'lm'.

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
static leastSquares(callable $fun, array $x0, ?callable $jac = null, float $ftol = 1.0E-8, float $xtol = 1.0E-8, float $gtol = 1.0E-8, ?int $maxNfev = null): Tessero\Optimize\OptimizeResult
```

scipy.optimize.least_squares with method 'lm' (unbounded): minimise 0.5 ||fun(x)||^2 from x0 with MINPACK's lmder
as SciPy 1.17 runs it, the Jacobian $jac's or approx_derivative's 2-point differences, max_nfev 100 n by default;
x, nfev, njev ($jac only), status and message follow SciPy's. fun is the cost (half the sum of squares,
summed pairwise, where SciPy uses its BLAS dot) and jac the Jacobian at x (one row per residual), as SciPy's jac.
SciPy's default method 'trf' takes other steps to the same minimum. Bounds and the trf/dogbox trust-region
methods are not supported.

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
static solve(callable $residuals, array $p0, ?callable $jac = null, float $ftol = 1.49012E-8, float $xtol = 1.49012E-8, float $gtol = 0.0, ?int $maxfev = null): array
```

scipy.optimize.leastsq: minimise sum(residuals(p)^2) with MINPACK's lmdif, or lmder with $jac, with leastsq's
defaults (maxfev 200 (n + 1), with $jac 100 (n + 1)), so x, the residuals, nfev and the status follow SciPy's.
status is leastsq's ier (1 .. 4: a solution found) and message its mesg; cost is half the sum of squares, cov_x
leastsq's cov_x (null when R is singular), jac the Jacobian at x ($jac's, else 2-point differences, which are
not counted in nfev).
