<?php

declare(strict_types=1);

namespace Tessero;

/**
 * scipy.ndimage.generic_filter, implemented in pure PHP (one class serves both backends, no kernel call).
 *
 * For every output element the footprint-selected neighbours are gathered in C order (row-major, the same
 * order scipy's C filter visits a footprint via np.argwhere), with the requested boundary fold applied per
 * axis, and the resulting length-K window is handed to the user callable. The callable returns one scalar.
 *
 * Element ORDER inside the window is load-bearing: order-sensitive callables (e.g. a weighted sum) see the
 * neighbours in exactly the order scipy would, so this matches scipy.ndimage.generic_filter bit-for-bit for
 * the arithmetic the callable performs.
 *
 * Boundary modes (reflect / constant / nearest / wrap / mirror) reproduce scipy's index fold; `constant`
 * substitutes `cval` for out-of-bounds samples. `origin` shifts the footprint centre (scalar or per-axis).
 *
 * Input is an N-D array given as PHP nested float lists; the result has the same shape.
 */
final class NdimageGeneric
{
    private function __construct() {}

    /**
     * @param array           $input     N-D nested float array (shape preserved in the result)
     * @param callable        $function  fn(array $window): float  -- window is a flat C-order float list
     * @param int|array|null  $size      scalar or per-axis filter size (ignored when $footprint is given)
     * @param array|null      $footprint N-D boolean/0-1 mask selecting the neighbourhood (overrides $size)
     * @param string          $mode      reflect|constant|nearest|wrap|mirror
     * @param float           $cval      fill value for out-of-bounds samples under mode=constant
     * @param int|array       $origin    scalar or per-axis footprint-centre shift
     *
     * @return array N-D nested float array, same shape as $input
     */
    public static function genericFilter(
        array $input,
        callable $function,
        int|array|null $size = null,
        ?array $footprint = null,
        string $mode = 'reflect',
        float $cval = 0.0,
        int|array $origin = 0,
    ): array {
        // --- shape of the (possibly nested) input ---
        $shape = self::shapeOf($input);
        $nd = count($shape);
        if ($nd === 0) {
            throw new \ValueError('generic_filter: input must be a non-empty array.');
        }
        $flat = self::flatten($input);                 // C-order flat values
        $strides = self::cStrides($shape);

        // --- footprint: either from size (full box) or the explicit mask ---
        if ($footprint === null) {
            if ($size === null) {
                throw new \ValueError('generic_filter: either size or footprint is required.');
            }
            $fshape = is_array($size) ? array_map('intval', array_values($size)) : array_fill(0, $nd, (int) $size);
            if (count($fshape) !== $nd) {
                throw new \ValueError('generic_filter: size length must match input rank.');
            }
            $fpFlat = array_fill(0, (int) array_product($fshape), true);
        } else {
            $fshape = self::shapeOf($footprint);
            if (count($fshape) !== $nd) {
                throw new \ValueError('generic_filter: footprint rank must match input rank.');
            }
            $fpFlat = array_map(static fn ($v) => (bool) $v, self::flatten($footprint));
        }
        $fstrides = self::cStrides($fshape);
        $center = [];
        foreach ($fshape as $s) {
            $center[] = intdiv($s, 2);
        }
        $orig = is_array($origin) ? array_map('intval', array_values($origin)) : array_fill(0, $nd, (int) $origin);
        if (count($orig) !== $nd) {
            throw new \ValueError('generic_filter: origin length must match input rank.');
        }

        // --- kept footprint offsets, in C order (matches scipy's np.argwhere ordering) ---
        $offs = [];                                    // each: per-axis offset vector
        $fcount = count($fpFlat);
        for ($fi = 0; $fi < $fcount; $fi++) {
            if (! $fpFlat[$fi]) {
                continue;
            }
            $rem = $fi;
            $off = [];
            foreach ($fstrides as $st) {
                $off[] = intdiv($rem, $st);
                $rem %= $st;
            }
            $offs[] = $off;
        }

        // --- per output element: gather the window (C order), apply the callable ---
        $total = count($flat);
        $out = array_fill(0, $total, 0.0);
        for ($oi = 0; $oi < $total; $oi++) {
            // decode output coordinate
            $rem = $oi;
            $idx = [];
            foreach ($strides as $st) {
                $idx[] = intdiv($rem, $st);
                $rem %= $st;
            }
            $window = [];
            foreach ($offs as $off) {
                $flatPos = 0;
                $oob = false;
                for ($d = 0; $d < $nd; $d++) {
                    $p = $idx[$d] + $off[$d] - $center[$d] - $orig[$d];
                    $sc = self::bmap($p, $shape[$d], $mode);
                    if ($sc < 0) {
                        $oob = true;
                        break;
                    }
                    $flatPos += $sc * $strides[$d];
                }
                $window[] = $oob ? $cval : $flat[$flatPos];
            }
            $out[$oi] = (float) $function($window);
        }

        return self::reshape($out, $shape);
    }

    /**
     * Boundary fold, identical to the C kernel's bmap (csrc/src/scipy_ndimage.c). Returns the in-bounds
     * source index for coordinate $p on an axis of length $n, or -1 for a constant-mode out-of-bounds sample.
     */
    private static function bmap(int $p, int $n, string $mode): int
    {
        if ($p >= 0 && $p < $n) {
            return $p;
        }
        switch ($mode) {
            case 'constant':
                return -1;
            case 'nearest':
                return $p < 0 ? 0 : $n - 1;
            case 'wrap':
                $p %= $n;
                if ($p < 0) {
                    $p += $n;
                }
                return $p;
            case 'mirror':                              // (d c b | a b c d | c b a), period 2n-2
                if ($n === 1) {
                    return 0;
                }
                $per = 2 * $n - 2;
                $p %= $per;
                if ($p < 0) {
                    $p += $per;
                }
                return $p < $n ? $p : $per - $p;
            case 'reflect':                             // (d c b a | a b c d | d c b a), period 2n
                $per = 2 * $n;
                $p %= $per;
                if ($p < 0) {
                    $p += $per;
                }
                return $p < $n ? $p : $per - 1 - $p;
            default:
                throw new \ValueError("generic_filter: unknown mode '{$mode}'.");
        }
    }

    /** C-order strides for a shape. */
    private static function cStrides(array $shape): array
    {
        $nd = count($shape);
        $strides = array_fill(0, $nd, 1);
        for ($d = $nd - 2; $d >= 0; $d--) {
            $strides[$d] = $strides[$d + 1] * $shape[$d + 1];
        }
        return $strides;
    }

    /** Shape of a (possibly nested) list. */
    private static function shapeOf(array $a): array
    {
        $shape = [];
        $cur = $a;
        while (is_array($cur)) {
            $shape[] = count($cur);
            $cur = $cur[array_key_first($cur)] ?? null;
        }
        return $shape;
    }

    /** C-order flatten of a nested list to a flat float list. */
    private static function flatten(array $a): array
    {
        $out = [];
        $walk = static function ($v) use (&$walk, &$out): void {
            if (is_array($v)) {
                foreach ($v as $e) {
                    $walk($e);
                }
            } else {
                $out[] = (float) $v;
            }
        };
        $walk($a);
        return $out;
    }

    /** Rebuild a nested list of the given shape from a flat C-order list. */
    private static function reshape(array $flat, array $shape): array
    {
        if (count($shape) === 1) {
            return array_values($flat);
        }
        $head = $shape[0];
        $rest = array_slice($shape, 1);
        $block = (int) array_product($rest);
        $out = [];
        for ($i = 0; $i < $head; $i++) {
            $out[] = self::reshape(array_slice($flat, $i * $block, $block), $rest);
        }
        return $out;
    }
}
