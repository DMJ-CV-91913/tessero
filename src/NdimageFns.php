<?php

/**
 * ===========================================================================
 * DELIVERABLE (1): impl -- pure-PHP generic_gradient_magnitude (standalone)
 * ===========================================================================
 *
 * TARGET: scipy.ndimage.generic_gradient_magnitude(input, derivative,
 *                                                   output=None, mode="reflect",
 *                                                   cval=0.0, ...)
 *
 * Gradient magnitude from a caller-supplied per-axis derivative callable:
 *     sqrt( sum_axis ( derivative(input, axis)^2 ) )
 *
 * Mirrors scipy exactly:
 *     out = derivative(input, 0)^2
 *     for ax in 1..ndim-1: out += derivative(input, ax)^2
 *     out = sqrt(out)
 * (rank-0 input returns a copy).
 *
 * PURE-PHP: no native routine of its own. It orchestrates NDArray ops
 * (square/add/sqrt) and defers the actual derivative to the PHP callable,
 * which -- per the fixtures -- is Ndimage::correlate1d (the canonical
 * derivative). The callable receives ($input, $axis, $mode, $cval) and must
 * return an NDArray on the active backend.
 *
 * WHERE THIS LANDS: src/Ndimage.php is machine-generated ("Do not edit").
 * Pure-PHP standalone ndimage helpers belong in a hand-written companion
 * class, following Tessero\StatsFns / Tessero\SpecialSeq / Tessero\OrthoPoly.
 * Suggested home: src/NdimageFns.php (class Tessero\NdimageFns).
 */

declare(strict_types=1);

namespace Tessero;

/**
 * Pure-PHP scipy.ndimage helpers that orchestrate native ndimage primitives
 * but are not themselves single native routines.
 */
final class NdimageFns
{
    private function __construct()
    {
    }

    /**
     * scipy.ndimage.generic_gradient_magnitude.
     *
     * sqrt(sum over axes of derivative(input, axis)^2).
     *
     * @param  object   $input       NDArray on the active backend (Tessero\NDArray or Tessero\Ext\NDArray); any rank
     * @param  callable $derivative  derivative(NDArray $input, int $axis, string $mode, float $cval): NDArray
     *                               e.g. fn ($in,$ax,$m,$c) => Ndimage::correlate1d($in,[-1,0,1],$ax,$m,$c)
     * @param  string   $mode        boundary mode forwarded to $derivative
     * @param  float    $cval        constant value for mode="constant", forwarded to $derivative
     * @return object                 NDArray on the active backend, same shape as $input
     */
    public static function genericGradientMagnitude(
        object $input,
        callable $derivative,
        string $mode = "reflect",
        float $cval = 0.0
    ): object {
        $ndim = $input->ndim();

        // Degenerate rank-0: scipy copies input through unchanged.
        if ($ndim === 0) {
            return $input->add(0.0);
        }

        // Axis 0: derivative, then square -> running accumulator.
        $acc = ($derivative($input, 0, $mode, $cval))->square();

        // Remaining axes: derivative, square, accumulate.
        for ($axis = 1; $axis < $ndim; $axis++) {
            $t = ($derivative($input, $axis, $mode, $cval))->square();
            $acc = $acc->add($t);
        }

        // Magnitude.
        return $acc->sqrt();
    }

    /**
     * scipy.ndimage.generic_laplace.
     *
     * Sum over every axis of the caller-supplied second-derivative callable.
     * With $derivative2 = correlate1d(weights=[1,-2,1]) this equals
     * scipy.ndimage.laplace exactly.
     *
     * @param  object   $input       NDArray on the active backend (any rank >= 1)
     * @param  callable $derivative2 derivative2(object $input, int $axis, string $mode, float $cval): object
     *                               e.g. fn ($in,$ax,$m,$c) => Ndimage::correlate1d($in,[1,-2,1],$ax,$m,$c)
     * @param  string   $mode        boundary mode forwarded to $derivative2
     * @param  float    $cval        constant value for mode="constant", forwarded to $derivative2
     * @return object                 NDArray on the active backend, same shape as $input
     */
    public static function genericLaplace(
        object $input,
        callable $derivative2,
        string $mode = "reflect",
        float $cval = 0.0
    ): object {
        $ndim = $input->ndim();
        if ($ndim < 1) {
            throw new \InvalidArgumentException('generic_laplace: input must have at least one axis');
        }

        // Axis 0 seeds the accumulator; remaining axes add their 2nd derivative.
        $out = $derivative2($input, 0, $mode, $cval);
        for ($axis = 1; $axis < $ndim; $axis++) {
            $out = $out->add($derivative2($input, $axis, $mode, $cval));
        }

        return $out;
    }
}
