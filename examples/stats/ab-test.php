<?php

/**
 * A/B test: a two-sample t-test on a per-user metric. Both backends.
 * SciPy: scipy.stats.ttest_ind(a, b)  ->  statistic, pvalue
 *
 * Scenario: revenue-per-visitor for two site variants. The t-test says whether the means
 * differ more than sampling noise would explain. Stats methods take plain PHP arrays.
 */

declare(strict_types=1);

$variantA = [5.1, 4.9, 5.2, 5.0, 5.3];
$variantB = [5.5, 5.6, 5.4, 5.7, 5.5];

$res = ST::ttestInd($variantA, $variantB);          // Student's t-test (equal variances, the default)

say(sprintf('t = %.6f  p = %.6g  on %s', $res['statistic'], $res['pvalue'], backend()));

check_close($res['statistic'], -5.047146145152368, 't statistic matches SciPy', 1e-9, 1e-12);
check_close($res['pvalue'], 0.0009927667987489237, 'p-value matches SciPy', 1e-9, 1e-15);
check($res['pvalue'] < 0.05, 'the variants differ significantly');
check($res['statistic'] < 0.0, 'variant A mean is below variant B');
