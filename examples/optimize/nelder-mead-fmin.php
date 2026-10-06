<?php

/**
 * Derivative-free minimisation with the Nelder-Mead simplex via fmin.
 * SciPy: scipy.optimize.fmin(func, x0)  (returns the minimiser)
 *
 * Scenario: a robust "typical value" for a sample with an outlier. Minimising the sum of
 * absolute deviations (an L1 loss, non-smooth, no usable gradient) gives the median.
 */

declare(strict_types=1);

use Tessero\Optimize\Minimize;

ffi_only('scipy.optimize');

$sample = [1.0, 2.0, 2.0, 3.0, 100.0];                 // median is 2.0, mean is distorted by 100
$l1 = static fn (array $x): float => array_sum(array_map(static fn ($d): float => abs($d - $x[0]), $sample));

$xmin = Minimize::fmin($l1, [0.0]);                    // fmin returns the minimiser as a list

say('L1 minimiser (median) = ' . json_encode($xmin));

check_close($xmin, [2.0], 'Nelder-Mead finds the median', 1e-3, 1e-3);
