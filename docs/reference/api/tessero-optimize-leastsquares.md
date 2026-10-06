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

### solve

```php
static solve(callable $residuals, array $p0, ?callable $jac = null, float $ftol = 1.0E-8, float $xtol = 1.0E-8, float $gtol = 1.0E-8, ?int $maxfev = null): array
```

Minimise 0.5 * sum(residuals(p)^2).
