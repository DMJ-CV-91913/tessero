<?php

declare(strict_types=1);

namespace Tessero;

/**
 * scipy.ndimage.vectorized_filter, pure PHP (one class serves both backends, no kernel call).
 *
 * vectorized_filter is the vectorised sibling of generic_filter: for every output element the
 * footprint-selected neighbours are gathered and handed to a callable. Where generic_filter's callable
 * takes one flat window and returns a scalar, vectorized_filter's callable has the signature
 *
 *     function(window: ndarray, *, axis: int | tuple) -> scalar
 *
 * and reduces the window over the given axis/axes. For a reducer that honours `axis`, evaluating one
 * window at a time (axis = 0 over the flat length-K window) yields exactly the same result scipy
 * produces for its batched evaluation — the reduction is the same arithmetic over the same K values.
 *
 * This implementation therefore delegates the window gather, boundary folding and footprint handling to
 * Tessero\NdimageGeneric (which reproduces scipy's C-order footprint visit order bit-for-bit) and wraps
 * the user's `(window, axis)` callable so NdimageGeneric can call it with the single flat window.
 *
 * Supported boundary modes: reflect|constant|nearest|wrap|mirror. `origin` shifts the footprint centre.
 * `cval` defaults to 0.0 (scipy uses None meaning 0.0 for the constant fill).
 */
final class VectorizedFilter
{
    private function __construct() {}

    /**
     * @param array           $input     N-D nested float array (shape preserved in the result)
     * @param callable        $function  fn(array $window, int|array $axis): float  (axis is always 0 here)
     * @param int|array|null  $size      scalar or per-axis filter size (ignored when $footprint is given)
     * @param array|null      $footprint N-D boolean/0-1 mask selecting the neighbourhood (overrides $size)
     * @param string          $mode      reflect|constant|nearest|wrap|mirror
     * @param float|null      $cval      fill value for out-of-bounds samples under mode=constant
     * @param int|array|null  $origin    scalar or per-axis footprint-centre shift
     *
     * @return array N-D nested float array, same shape as $input
     */
    public static function vectorizedFilter(
        array $input,
        callable $function,
        int|array|null $size = null,
        ?array $footprint = null,
        string $mode = 'reflect',
        ?float $cval = null,
        int|array|null $origin = null,
    ): array {
        // The window is a flat length-K list; present it as a single reduction axis (axis 0),
        // which is what scipy's vectorised callables reduce over for a per-element window.
        $wrapped = static fn (array $window): float => (float) $function($window, 0);

        return NdimageGeneric::genericFilter(
            $input,
            $wrapped,
            $size,
            $footprint,
            $mode,
            $cval ?? 0.0,
            $origin ?? 0,
        );
    }
}
