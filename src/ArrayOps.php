<?php

declare(strict_types=1);

namespace Tessero;

/**
 * Array-assembly helpers that are not single kernel routines.
 *
 * The native extension shares this class: block() operates on NDArray objects of either backend
 * (Tessero\NDArray or Tessero\Ext\NDArray) and routes concatenation through the matching facade.
 */
final class ArrayOps
{
    private function __construct()
    {
    }

    /**
     * Assemble a nested list of blocks into a single array (numpy.block).
     *
     * Innermost lists are concatenated along the last axis, the next level out along the
     * second-to-last axis, and so on; leaf arrays of lower rank are promoted by prepending
     * length-one axes, exactly as NumPy does.
     *
     * @param array<mixed> $arrays a (possibly nested) list whose leaves are NDArray|Ext\NDArray or numbers
     */
    public static function block(array $arrays): mixed
    {
        $depth = self::listDepth($arrays);
        $ndim = max($depth, self::maxLeafNdim($arrays));
        $cls = self::firstObjectClass($arrays);
        $ext = $cls !== null && str_contains($cls, '\\Ext\\');
        $arrClass = $ext ? 'Tessero\\Ext\\NDArray' : 'Tessero\\NDArray';
        $npClass = $ext ? 'Tessero\\Ext\\Np' : 'Tessero\\Np';

        return self::assemble($arrays, $depth, $ndim, $arrClass, $npClass);
    }

    /** nesting depth of the list structure (a leaf has depth 0). */
    private static function listDepth(mixed $node): int
    {
        if (! is_array($node)) {
            return 0;
        }
        $m = 0;
        foreach ($node as $c) {
            $d = self::listDepth($c);
            if ($d > $m) {
                $m = $d;
            }
        }

        return $m + 1;
    }

    /** the largest rank among the leaf arrays (a scalar leaf counts as rank 0). */
    private static function maxLeafNdim(mixed $node): int
    {
        if (is_array($node)) {
            $m = 0;
            foreach ($node as $c) {
                $d = self::maxLeafNdim($c);
                if ($d > $m) {
                    $m = $d;
                }
            }

            return $m;
        }

        return is_object($node) ? count($node->shape()) : 0;
    }

    /** the class of the first NDArray leaf, used to pick the backend (null when there is none). */
    private static function firstObjectClass(mixed $node): ?string
    {
        if (is_object($node)) {
            return $node::class;
        }
        if (is_array($node)) {
            foreach ($node as $c) {
                $r = self::firstObjectClass($c);
                if ($r !== null) {
                    return $r;
                }
            }
        }

        return null;
    }

    private static function assemble(mixed $node, int $depth, int $ndim, string $arrClass, string $npClass): mixed
    {
        if (! is_array($node)) {
            return self::atLeast($node, $ndim, $arrClass);
        }
        $parts = [];
        foreach ($node as $c) {
            $parts[] = self::assemble($c, $depth - 1, $ndim, $arrClass, $npClass);
        }

        return $npClass::concatenate($parts, $ndim - $depth);
    }

    /** promote a leaf (array or scalar) to $ndim axes by prepending length-one axes. */
    private static function atLeast(mixed $leaf, int $ndim, string $arrClass): mixed
    {
        if (! is_object($leaf)) {
            $leaf = is_int($leaf) ? $arrClass::array([$leaf], 'int64') : $arrClass::array([(float) $leaf]);
        }
        $shape = $leaf->shape();
        $cur = count($shape);
        if ($cur < $ndim) {
            return $leaf->reshape(array_merge(array_fill(0, $ndim - $cur, 1), $shape));
        }

        return $leaf;
    }
}
