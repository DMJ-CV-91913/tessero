# Optimisation, roots and curve fitting

These follow `scipy.optimize`. The objective functions are PHP callables, so
they are driven from PHP; the linear algebra inside them runs natively. For
linear and mixed-integer programs, see [Linear programming](linear-programming.md).
Those solvers run entirely in C.

## minimize

```php
use Tessero\Optimize\Minimize;

$rosen = fn (array $x): float => 100 * ($x[1] - $x[0] ** 2) ** 2 + (1 - $x[0]) ** 2;

$r = Minimize::minimize($rosen, [-1.2, 1.0]);                        // BFGS
$r = Minimize::minimize($rosen, [-1.2, 1.0], method: 'Nelder-Mead');
$r = Minimize::minimize($rosen, [-1.2, 1.0], method: 'L-BFGS-B', bounds: [[-2, 2], [-2, 2]]);
$r = Minimize::minimize($rosen, [-1.2, 1.0], jac: $gradient);        // analytic gradient

$r->x;        // [1.0, 1.0] (approximately)
$r->fun; $r->success; $r->message; $r->nit; $r->nfev;
```

| Method | Notes |
|---|---|
| `BFGS` (default) | SciPy's algorithm, including its Wolfe line search; same iterates as SciPy on the test problems |
| `L-BFGS-B` | projected L-BFGS with bound constraints; same optimum as SciPy, different path |
| `Nelder-Mead` | same iterates as SciPy (`adaptive` option supported) |

Options go in `options: ['maxiter' => …, 'gtol' => …, 'xatol' => …, 'fatol' => …]`
with SciPy's names. Without `jac` the gradient is approximated by forward
differences, using SciPy's step size.

## Roots

```php
use Tessero\Optimize\Root;

Root::brentq(fn ($x) => $x ** 3 - 2 * $x - 5, 2, 3);     // 2.094551481542327
Root::newton(fn ($x) => cos($x) - $x, 1.0);              // secant method without fprime
Root::root($F, [1.0, 1.0]);                               // systems: damped Newton
```

`brentq` gives the same root and the same number of function calls as SciPy.

## Least squares and curve fitting

```php
use Tessero\Optimize\LeastSquares;

$model = fn (float $t, float $a, float $k): float => $a * exp(-$k * $t);
[$popt, $pcov] = LeastSquares::curveFit($model, $t, $y, p0: [1.0, 0.1]);
$stderr = array_map('sqrt', [$pcov[0][0], $pcov[1][1]]);

LeastSquares::solve($residuals, $p0);    // Levenberg-Marquardt on a residual function
```

`curveFit` scales the covariance as SciPy does, and supports `sigma` and
`absoluteSigma`.

## Limits

There are no trust-region methods and no general nonlinear constraints (only
bounds, through L-BFGS-B). Use the [LP/MILP solvers](linear-programming.md) for
linear constraints.
