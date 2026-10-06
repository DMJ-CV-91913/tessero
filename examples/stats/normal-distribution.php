<?php

/**
 * Work with a distribution object: pdf/cdf/ppf of the standard normal. Both backends.
 * SciPy: scipy.stats.norm(0, 1).cdf(1.96); .ppf(0.975)
 *
 * Scenario: the z-score behind a 95% confidence interval. cdf(1.96) ~= 0.975, and ppf is its
 * inverse, so ppf(0.975) ~= 1.96.
 */

declare(strict_types=1);

$z = ST::norm(0.0, 1.0);

check_close($z->cdf(1.96), 0.9750021048517795, 'Phi(1.96) ~= 0.975', 1e-9, 1e-12);
check_close($z->ppf(0.975), 1.959963984540054, 'the 97.5% quantile ~= 1.96', 1e-9, 1e-12);
check_close($z->pdf(0.0), 1.0 / sqrt(2.0 * M_PI), 'peak density at 0', 1e-12, 1e-12);

// A vector of quantiles in one call.
check_close($z->cdf([-1.0, 0.0, 1.0])->toList(), [0.15865525393145707, 0.5, 0.8413447460685429], 'cdf vector', 1e-9, 1e-12);

say('standard-normal pdf/cdf/ppf ok on ' . backend());
