<?php

/**
 * Two workhorse special functions: the logistic sigmoid and the error function. Both backends.
 * SciPy: scipy.special.expit(x); scipy.special.erf(x)
 *
 * expit(x) = 1 / (1 + e^-x) is the logistic link used in classification; erf underlies the
 * normal CDF. Scalars return a float; arrays return an NDArray.
 */

declare(strict_types=1);

check_close(SP::expit(0.0), 0.5, 'expit(0) = 0.5', 1e-12, 1e-12);
check_close(
    SP::expit([-2.0, 0.0, 2.0])->toList(),
    [0.11920292202211755, 0.5, 0.8807970779778823],
    'expit over a vector',
    1e-12,
    1e-12,
);

check_close(SP::erf(1.0), 0.8427007929497149, 'erf(1)', 1e-12, 1e-12);
check_close(SP::erf(0.0), 0.0, 'erf(0) = 0', 1e-12, 1e-12);

say('expit and erf ok on ' . backend());
