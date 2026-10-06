<?php

/**
 * Bracketed root finding with Brent's method.
 * SciPy: scipy.optimize.brentq(f, a, b)
 *
 * Scenario: the break-even price. Demand falls with price, q(p) = 1000 * exp(-p / 50);
 * profit(p) = (p - unit_cost) * q(p) - fixed_cost. brentq finds the price where profit = 0.
 */

declare(strict_types=1);

use Tessero\Optimize\Root;

ffi_only('scipy.optimize');

$fixedCost = 2000.0;
$unitCost = 10.0;
$profit = static function (float $p) use ($fixedCost, $unitCost): float {
    $q = 1000.0 * exp(-$p / 50.0);

    return ($p - $unitCost) * $q - $fixedCost;
};

$price = Root::brentq($profit, 11.0, 200.0);           // profit is negative at 11, positive at 200

say('break-even price = ' . round($price, 4));

check(abs($profit($price)) < 1e-6, 'profit at the break-even price should be ~0');
check($price > 11.0 && $price < 200.0, 'root lies inside the bracket');
