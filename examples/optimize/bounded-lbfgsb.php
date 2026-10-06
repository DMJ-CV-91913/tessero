<?php

/**
 * Box-constrained minimisation. Passing `bounds` switches minimize to L-BFGS-B automatically.
 * SciPy: scipy.optimize.minimize(f, x0, method='L-BFGS-B', bounds=[(-2, 0.5), (None, None)])
 *
 * Scenario: pick two control settings that minimise a smooth cost, where the first setting
 * is physically capped at 0.5. The unconstrained optimum is (3, 2); the cap pins x0 to 0.5.
 */

declare(strict_types=1);

use Tessero\Optimize\Minimize;

ffi_only('scipy.optimize');

$cost = static fn (array $x): float => ($x[0] - 3.0) ** 2 + ($x[1] - 2.0) ** 2;

// null as a bound means unbounded on that side (SciPy's None).
$res = Minimize::minimize($cost, [-1.2, 1.0], bounds: [[-2.0, 0.5], [null, null]]);

say('x = ' . json_encode($res->x) . '  method = ' . $res->method);

check($res->success, 'L-BFGS-B should converge');
check_close($res->x, [0.5, 2.0], 'x0 pinned to its upper bound, x1 free at the optimum', 1e-4, 1e-5);
