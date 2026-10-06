<?php

/**
 * Combinatorics with the gamma family. Both backends.
 * SciPy: scipy.special.comb(n, k); scipy.special.gammaln(x)
 *
 * comb is the binomial coefficient; gammaln(n+1) = log(n!), the numerically safe way to work
 * with factorials that would otherwise overflow.
 */

declare(strict_types=1);

check_close(SP::comb(10, 3), 120.0, 'C(10, 3) = 120', 1e-9, 1e-9);
check_close(SP::comb(52, 5), 2598960.0, 'poker hands C(52, 5)', 1e-6, 1e-3);

// gammaln(6) = log(5!) = log(120)
check_close(SP::gammaln(6.0), log(120.0), 'gammaln(6) = log(5!)', 1e-12, 1e-12);
// log of a large factorial that would overflow as a plain float factorial
check_close(SP::gammaln(101.0), 363.73937555556347, 'gammaln(101) = log(100!)', 1e-9, 1e-9);

say('comb and gammaln ok on ' . backend());
