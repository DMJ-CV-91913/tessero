<?php

/**
 * Unconstrained minimisation with quasi-Newton BFGS.
 * SciPy: scipy.optimize.minimize(rosen, [-1.2, 1.0], method='BFGS')
 *
 * The Rosenbrock function is the classic optimiser stress test: a curved valley whose
 * minimum is at (1, 1) with value 0.
 */

declare(strict_types=1);

use Tessero\Optimize\Minimize;

ffi_only('scipy.optimize');

$rosen = static fn (array $x): float => 100.0 * ($x[1] - $x[0] ** 2) ** 2 + (1.0 - $x[0]) ** 2;

$res = Minimize::minimize($rosen, [-1.2, 1.0], method: 'BFGS');

say('x = ' . json_encode($res->x) . '  f = ' . $res->fun . '  iters = ' . $res->nit);

check($res->success, 'BFGS should converge on Rosenbrock');
check_close($res->x, [1.0, 1.0], 'minimiser should be (1, 1)', 1e-4, 1e-5);
check($res->fun < 1e-8, 'objective at the minimum should be ~0');
