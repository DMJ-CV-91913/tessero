<?php

/**
 * Ordinary least-squares line fit with correlation and significance. Both backends.
 * SciPy: scipy.stats.linregress(x, y)  ->  slope, intercept, rvalue, pvalue, stderr, ...
 *
 * Scenario: a simple trend - spend vs. week - to forecast and judge how strong the fit is.
 */

declare(strict_types=1);

$week = [1, 2, 3, 4, 5];
$spend = [2.1, 3.9, 6.1, 8.0, 9.9];

$fit = ST::linregress($week, $spend);

say(sprintf('spend ~= %.3f * week + %.3f  (r = %.4f)  on %s', $fit['slope'], $fit['intercept'], $fit['rvalue'], backend()));

check_close($fit['slope'], 1.9700000000000004, 'slope matches SciPy', 1e-9, 1e-12);
check_close($fit['intercept'], 0.08999999999999897, 'intercept matches SciPy', 1e-9, 1e-9);
check_close($fit['rvalue'], 0.9996008472180341, 'correlation matches SciPy', 1e-9, 1e-12);
check($fit['pvalue'] < 0.001, 'the trend is highly significant');
