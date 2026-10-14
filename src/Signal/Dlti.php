<?php

declare(strict_types=1);

namespace Tessero\Signal;

/**
 * scipy.signal.dlti - discrete-time LTI system factory base.
 *
 * Mirrors scipy: `dlti(*system, dt=True)` dispatches on the number of
 * positional arguments to the discrete flavour of each representation
 * (TransferFunction / ZerosPolesGain / StateSpace carrying a non-null dt).
 * Discrete responses use the dimpulse / dstep / dbode primitives (inherited
 * from Lti). The default sampling interval in scipy is dt=True (== 1.0).
 */
final class Dlti
{
    /**
     * Factory: construct the discrete representation from a variable argument list
     * plus a sampling interval dt (keyword in scipy; trailing argument here).
     *
     * @return TransferFunction|ZerosPolesGain|StateSpace
     */
    public static function create(array $system, float $dt = 1.0): Lti
    {
        switch (count($system)) {
            case 2:
                return new TransferFunction($system[0], $system[1], $dt);
            case 3:
                return new ZerosPolesGain($system[0], $system[1], $system[2], $dt);
            case 4:
                return new StateSpace($system[0], $system[1], $system[2], $system[3], $dt);
            default:
                throw new \InvalidArgumentException(
                    'dlti: needs 2 (num,den), 3 (z,p,k) or 4 (A,B,C,D) arguments, got ' . count($system) . '.'
                );
        }
    }
}
