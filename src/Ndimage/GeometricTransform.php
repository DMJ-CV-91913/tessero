<?php

declare(strict_types=1);

namespace Tessero\Ndimage;

use Tessero\Ndimage;

/**
 * scipy.ndimage.geometric_transform: apply an arbitrary coordinate mapping to an array.
 *
 * This is a pure-PHP carve-out (hand-written, not part of the generated {@see \Tessero\Ndimage}
 * facade). SciPy's C routine walks every output coordinate, calls the user mapping callable to get
 * the corresponding input coordinate, and B-spline interpolates the input there. We do exactly that
 * on top of the existing {@see \Tessero\Ndimage::mapCoordinates} routine: build the full coordinates
 * array (shape (ndim, *output_shape)) by calling the mapping once per output cell, then delegate a
 * single map_coordinates call -- which already prefilters and interpolates orders 0-5 under every
 * boundary mode. Identical on both backends (the only numeric kernel is map_coordinates).
 *
 * The mapping is a PHP callable: given an output coordinate as a list<int> (length = input rank),
 * it returns the input coordinate as a list<float> (length = input rank). This matches scipy's
 * `mapping(output_coordinates) -> input_coordinates` contract (extra_arguments/extra_keywords are
 * handled by the caller binding them into the closure).
 */
final class GeometricTransform
{
    private function __construct()
    {
    }

    /**
     * @param mixed                     $input       input array (NDArray | nested PHP array), rank >= 1
     * @param callable                  $mapping     list<int> output coord -> list<float> input coord
     * @param list<int>|null            $outputShape output shape; defaults to the input shape
     * @param int                       $order       spline order 0..5
     * @param string                    $mode        boundary mode (constant|nearest|reflect|mirror|wrap|grid-*)
     * @param float                     $cval        fill value for mode 'constant'
     * @param bool                      $prefilter   run the B-spline prefilter (orders > 1)
     */
    public static function apply(
        mixed $input,
        callable $mapping,
        ?array $outputShape = null,
        int $order = 3,
        string $mode = 'constant',
        float $cval = 0.0,
        bool $prefilter = true,
    ): mixed {
        // BOTH-BACKEND FIX: map_coordinates below is the FFI Ndimage facade (one numeric kernel, used on
        // both backends like the other pure-PHP carve-outs). On the ext backend $input arrives as a
        // Tessero\Ext\NDArray, which the FFI facade rejects. Duck-type any NDArray (toArray()) to a nested
        // PHP array first, so the single kernel accepts it identically under FFI and the extension.
        if (is_object($input) && method_exists($input, 'toArray')) {
            $input = $input->toArray();
        }
        $inShape = self::shapeOf($input);
        $ndim = count($inShape);
        if ($ndim === 0) {
            throw new \ValueError('geometric_transform: input must have rank >= 1');
        }
        $outShape = $outputShape ?? $inShape;
        if (count($outShape) !== $ndim) {
            throw new \ValueError('geometric_transform: output_shape rank must equal input rank');
        }

        // Build coordinates array of shape (ndim, *outShape): coords[d] is the input d-coordinate for
        // every output cell. We fill a flat buffer per axis in row-major (C) order over outShape, then
        // fold each axis buffer back into the nested output shape, matching map_coordinates' layout.
        $total = 1;
        foreach ($outShape as $s) {
            $total *= $s;
        }

        /** @var list<list<float>> $flat  flat[d][k] = input coord d for the k-th output cell */
        $flat = array_fill(0, $ndim, array_fill(0, $total, 0.0));
        $out = array_fill(0, $ndim, 0);                 // current output coordinate (odometer)
        for ($k = 0; $k < $total; $k++) {
            $src = $mapping($out);                      // list<float>, length ndim
            if (! is_array($src) || count($src) !== $ndim) {
                throw new \ValueError('geometric_transform: mapping must return ' . $ndim . ' coordinates');
            }
            $d = 0;
            foreach ($src as $v) {
                $flat[$d][$k] = (float) $v;
                $d++;
            }
            // increment the odometer (last axis fastest = C order)
            for ($ax = $ndim - 1; $ax >= 0; $ax--) {
                if (++$out[$ax] < $outShape[$ax]) {
                    break;
                }
                $out[$ax] = 0;
            }
        }

        $coords = [];
        for ($d = 0; $d < $ndim; $d++) {
            $coords[$d] = self::fold($flat[$d], $outShape);
        }

        return Ndimage::mapCoordinates($input, $coords, $order, $mode, $cval, $prefilter);
    }

    /**
     * Shape of an NDArray or a nested PHP array.
     *
     * @return list<int>
     */
    private static function shapeOf(mixed $a): array
    {
        if (is_object($a) && method_exists($a, 'shape')) {
            return array_map('intval', $a->shape());
        }
        $shape = [];
        $cur = $a;
        while (is_array($cur)) {
            $shape[] = count($cur);
            $cur = $cur[array_key_first($cur)] ?? null;
        }

        return $shape;
    }

    /**
     * Fold a flat (C-order) list into a nested array of the given shape.
     *
     * @param list<float> $flat
     * @param list<int>   $shape
     */
    private static function fold(array $flat, array $shape): array
    {
        if (count($shape) <= 1) {
            return $flat;
        }
        $head = $shape[0];
        $rest = array_slice($shape, 1);
        $block = 1;
        foreach ($rest as $s) {
            $block *= $s;
        }
        $out = [];
        for ($i = 0; $i < $head; $i++) {
            $out[$i] = self::fold(array_slice($flat, $i * $block, $block), $rest);
        }

        return $out;
    }
}
