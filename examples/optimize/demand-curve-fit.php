<?php

/**
 * Nonlinear curve fitting.
 * SciPy: scipy.optimize.curve_fit(model, xdata, ydata, p0)  ->  (params, covariance)
 * Tessero: Tessero\Optimize\LeastSquares::curveFit(...); the model is called model(x, ...params).
 *
 * Scenario: fit a demand curve q(p) = a * exp(-b * p) to observed (price, quantity) points
 * and recover the elasticity parameters a and b.
 */

declare(strict_types=1);

use Tessero\Optimize\LeastSquares;

ffi_only('scipy.optimize');

$model = static fn (float $p, float ...$c): float => $c[0] * exp(-$c[1] * $p);

$prices = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
$quantity = array_map(static fn (float $p): float => 1000.0 * exp(-0.02 * $p), $prices);   // true a=1000, b=0.02

[$params, $cov] = LeastSquares::curveFit($model, $prices, $quantity, [500.0, 0.01]);

say('a = ' . round($params[0], 4) . '  b = ' . round($params[1], 6));

check_close($params, [1000.0, 0.02], 'recover the demand parameters', 1e-4, 1e-6);
check(count($cov) === 2 && count($cov[0]) === 2, 'covariance is 2x2');
