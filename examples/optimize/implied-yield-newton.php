<?php

/**
 * Newton's method for a scalar root (no bracket needed, uses a numerical derivative here).
 * SciPy: scipy.optimize.newton(f, x0)
 *
 * Scenario: the implied annual yield of a bond. Find the rate r at which the present value
 * of the coupon/redemption cashflows equals the market price.
 */

declare(strict_types=1);

use Tessero\Optimize\Root;

ffi_only('scipy.optimize');

$cashflows = [5.0, 5.0, 5.0, 105.0];                   // 5% annual coupon, par 100, 4 years
$price = 98.0;
$pvGap = static function (float $r) use ($cashflows, $price): float {
    $pv = 0.0;
    foreach ($cashflows as $t => $cf) {
        $pv += $cf / (1.0 + $r) ** ($t + 1);
    }

    return $pv - $price;
};

$rate = Root::newton($pvGap, 0.05);                    // start from a 5% guess

say('implied annual yield = ' . round($rate * 100, 4) . '%');

check(abs($pvGap($rate)) < 1e-8, 'present value at the implied rate equals the price');
check($rate > 0.05 && $rate < 0.07, 'a bond priced below par yields a little over its coupon');
