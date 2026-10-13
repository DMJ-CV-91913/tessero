<?php

declare(strict_types=1);

namespace Tessero\Ndimage;

/**
 * scipy.ndimage.generic_filter1d: slide a user-supplied 1-D filter along one axis of an array.
 *
 * For every line taken along `axis`, SciPy builds an *extended* line of length n + filter_size - 1
 * (the real samples flanked by boundary-folded samples), then calls the user function ONCE for that
 * line with two buffers:
 *
 *     function(array $inbuf, int $outlen): array        // returns $outlen values
 *
 * where `$inbuf` is the extended line (length n + filter_size - 1) and the return is the output line
 * (length n). This mirrors SciPy's C signature `callback(input_line, output_line)`: the callee reads the
 * extended input and writes `n` outputs. We hand the function the extended buffer and the expected output
 * length and take its return array as the output line (a PHP-idiomatic equivalent of the out-parameter).
 *
 * The extended line is assembled exactly as SciPy's `NI_ExtendLine`:
 *   left  = filter_size // 2 + origin        (samples prepended before index 0)
 *   right = (filter_size - 1) - left         (samples appended after index n-1)
 * each out-of-range position folded through the requested boundary mode (constant / nearest / wrap /
 * mirror / reflect). This is the SAME fold as the kernel's `windowed_*` filters (bmap).
 *
 * Pure PHP (the kernel is never touched) -> one class, identical on both the FFI and ext backends.
 */
final class GenericFilter1d
{
    /**
     * @param array<int,mixed> $input      1-D list<float> or an n-D nested (row-major) array
     * @param callable         $function   fn(list<float> $inbuf, int $outlen): list<float>
     * @param int              $filterSize length of the filter window (>= 1)
     * @param int              $axis       axis to filter along (negative counts from the end)
     * @param string           $mode       reflect | constant | nearest | mirror | wrap
     * @param float            $cval       fill value for 'constant' out-of-range positions
     * @param int              $origin     shift of the window centre (SciPy's `origin`)
     *
     * @return array the filtered array, same shape as $input
     */
    public static function apply(
        array $input,
        callable $function,
        int $filterSize,
        int $axis = -1,
        string $mode = 'reflect',
        float $cval = 0.0,
        int $origin = 0,
    ): array {
        if ($filterSize < 1) {
            throw new \InvalidArgumentException('filter_size must be >= 1');
        }
        $shape = self::shape($input);
        $nd = count($shape);
        $axis = self::normAxis($axis, $nd);
        $n = $shape[$axis];

        // left/right padding exactly as scipy NI_ExtendLine (origin shifts the window centre)
        $left = intdiv($filterSize, 2) + $origin;
        $right = ($filterSize - 1) - $left;
        if ($left < 0 || $right < 0) {
            throw new \InvalidArgumentException('origin out of range for filter_size');
        }

        // Flatten to a nested structure we can address by multi-index, filter each line, rebuild.
        $out = self::mapLines($input, $shape, $axis, static function (array $line) use (
            $function, $filterSize, $mode, $cval, $left, $right, $n
        ): array {
            $ext = self::extendLine($line, $filterSize, $mode, $cval, $left, $right);
            $res = $function($ext, $n);
            $res = array_values(array_map('floatval', (array) $res));
            if (count($res) !== $n) {
                throw new \RuntimeException('filter returned ' . count($res) . " values, expected {$n}");
            }

            return $res;
        });

        return $out;
    }

    /**
     * Build the extended line: [left folded samples] ++ line ++ [right folded samples].
     * length = n + filter_size - 1  (= left + n + right).
     *
     * @param list<float> $line
     *
     * @return list<float>
     */
    private static function extendLine(array $line, int $filterSize, string $mode, float $cval, int $left, int $right): array
    {
        $n = count($line);
        $ext = [];
        for ($p = -$left; $p < $n + $right; $p++) {
            $sc = self::bmap($p, $n, $mode);
            $ext[] = $sc < 0 ? $cval : (float) $line[$sc];
        }

        return $ext;
    }

    /**
     * Boundary fold: map an out-of-range position `p` into [0, n) per `mode`, or return -1 to signal a
     * 'constant' fill. Mirrors the kernel's bmap and scipy's map_coordinate for the five base modes.
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
            case 'mirror':   // (d c b | a b c d | c b a), period 2n-2
                if ($n === 1) {
                    return 0;
                }
                $per = 2 * $n - 2;
                $p %= $per;
                if ($p < 0) {
                    $p += $per;
                }

                return $p < $n ? $p : $per - $p;
            case 'reflect':  // (d c b a | a b c d | d c b a), period 2n
                $per = 2 * $n;
                $p %= $per;
                if ($p < 0) {
                    $p += $per;
                }

                return $p < $n ? $p : $per - 1 - $p;
            default:
                throw new \InvalidArgumentException("unknown mode '{$mode}'");
        }
    }

    /**
     * Apply `$fn` (line -> line, same length) to every 1-D line of `$input` taken along `$axis`.
     *
     * @param array<int,mixed> $input
     * @param list<int>        $shape
     *
     * @return array<int,mixed>
     */
    private static function mapLines(array $input, array $shape, int $axis, callable $fn): array
    {
        $nd = count($shape);
        if ($nd === 1) {
            return $fn(array_map('floatval', array_values($input)));
        }

        // iterate every multi-index of the non-axis dimensions, pull the line, filter, write back
        $out = self::zerosLike($shape);
        $outerShape = $shape;
        unset($outerShape[$axis]);
        $outerShape = array_values($outerShape);

        foreach (self::ndindex($outerShape) as $outer) {
            // reconstruct the full index with a placeholder at $axis, then sweep the axis
            $line = [];
            for ($k = 0; $k < $shape[$axis]; $k++) {
                $idx = self::insertAt($outer, $axis, $k);
                $line[] = (float) self::getAt($input, $idx);
            }
            $res = $fn($line);
            for ($k = 0; $k < $shape[$axis]; $k++) {
                $idx = self::insertAt($outer, $axis, $k);
                self::setAt($out, $idx, $res[$k]);
            }
        }

        return $out;
    }

    /** @return list<int> */
    private static function shape(array $a): array
    {
        $shape = [];
        $cur = $a;
        while (is_array($cur)) {
            $shape[] = count($cur);
            $cur = $cur[0] ?? null;
        }

        return $shape;
    }

    private static function normAxis(int $axis, int $nd): int
    {
        if ($axis < 0) {
            $axis += $nd;
        }
        if ($axis < 0 || $axis >= $nd) {
            throw new \InvalidArgumentException('axis out of range');
        }

        return $axis;
    }

    /** @param list<int> $shape @return array<int,mixed> */
    private static function zerosLike(array $shape): array
    {
        if (count($shape) === 1) {
            return array_fill(0, $shape[0], 0.0);
        }
        $head = $shape[0];
        $rest = array_slice($shape, 1);
        $out = [];
        for ($i = 0; $i < $head; $i++) {
            $out[$i] = self::zerosLike($rest);
        }

        return $out;
    }

    /**
     * Generate every multi-index of `$shape` in row-major (C) order.
     *
     * @param list<int> $shape
     *
     * @return \Generator<int,list<int>>
     */
    private static function ndindex(array $shape): \Generator
    {
        if ($shape === []) {
            yield [];

            return;
        }
        $total = 1;
        foreach ($shape as $d) {
            $total *= $d;
        }
        for ($flat = 0; $flat < $total; $flat++) {
            $idx = [];
            $rem = $flat;
            for ($d = count($shape) - 1; $d >= 0; $d--) {
                $idx[$d] = $rem % $shape[$d];
                $rem = intdiv($rem, $shape[$d]);
            }
            ksort($idx);
            yield array_values($idx);
        }
    }

    /**
     * @param list<int> $outer index over the non-axis dims
     *
     * @return list<int> full index with `$val` spliced in at position `$axis`
     */
    private static function insertAt(array $outer, int $axis, int $val): array
    {
        array_splice($outer, $axis, 0, [$val]);

        return $outer;
    }

    /** @param array<int,mixed> $a @param list<int> $idx */
    private static function getAt(array $a, array $idx): mixed
    {
        $cur = $a;
        foreach ($idx as $i) {
            $cur = $cur[$i];
        }

        return $cur;
    }

    /** @param array<int,mixed> $a @param list<int> $idx */
    private static function setAt(array &$a, array $idx, float $val): void
    {
        $cur = &$a;
        $last = count($idx) - 1;
        foreach ($idx as $d => $i) {
            if ($d === $last) {
                $cur[$i] = $val;

                return;
            }
            $cur = &$cur[$i];
        }
    }
}
