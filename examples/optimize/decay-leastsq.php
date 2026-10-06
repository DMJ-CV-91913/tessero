<?php

/**
 * Nonlinear least squares from an explicit residual function (Levenberg-Marquardt).
 * SciPy: scipy.optimize.leastsq(residuals, p0)  /  least_squares(residuals, p0)
 * Tessero: Tessero\Optimize\LeastSquares::solve(residuals, p0) -> ['x'=>..., 'success'=>..., ...]
 *
 * Scenario: fit an exponential decay y = a * exp(b * t) (e.g. a cooling curve or churn) by
 * minimising the residual vector residuals(p)[i] = model(t_i, p) - y_i.
 */

declare(strict_types=1);

use Tessero\Optimize\LeastSquares;

ffi_only('scipy.optimize');

$t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
$y = array_map(static fn (float $ti): float => 3.0 * exp(-0.5 * $ti), $t);     // true a=3, b=-0.5

$residuals = static fn (array $p): array => array_map(
    static fn (float $ti, float $yi): float => $p[0] * exp($p[1] * $ti) - $yi,
    $t,
    $y,
);

$fit = LeastSquares::solve($residuals, [1.0, -0.1]);

say('a = ' . round($fit['x'][0], 4) . '  b = ' . round($fit['x'][1], 4) . '  cost = ' . $fit['cost']);

check($fit['success'], 'Levenberg-Marquardt should converge');
check_close($fit['x'], [3.0, -0.5], 'recover the decay parameters', 1e-4, 1e-6);
