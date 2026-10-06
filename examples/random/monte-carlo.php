<?php

/**
 * A seeded Monte-Carlo simulation, vectorised. Runs on both backends.
 * NumPy: g = np.random.default_rng(2024); x = g.uniform(size=n); ((x**2+y**2)<=1).mean()*4
 *
 * Estimate pi by sampling points in the unit square and counting those inside the quarter
 * circle. Seeded, so the estimate is deterministic and we can assert it is close to pi.
 */

declare(strict_types=1);

$g = RNG::defaultRng(2024);
$n = 100000;
$x = $g->uniform(0.0, 1.0, $n);
$y = $g->uniform(0.0, 1.0, $n);

$r2 = M::add(M::square($x), M::square($y));   // x^2 + y^2, element-wise
$inside = M::lessEqual($r2, 1.0)->sum();      // count of points inside the quarter circle
$piEst = 4.0 * (float) $inside / $n;

say('pi estimate (' . $n . ' samples, seed 2024) = ' . round($piEst, 5) . '  on ' . backend());
check(abs($piEst - M_PI) < 0.02, 'Monte-Carlo estimate is close to pi');
