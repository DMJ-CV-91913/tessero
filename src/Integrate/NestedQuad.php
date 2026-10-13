<?php

declare(strict_types=1);

namespace Tessero\Integrate;

/**
 * Nested adaptive Gauss-Kronrod quadrature over a user callable (scipy.integrate.dblquad / tplquad / nquad).
 *
 * scipy implements these exactly as nested one-dimensional QUADPACK `quad` calls: the outermost variable is
 * integrated by the adaptive GK15 rule, and each evaluation of the outer integrand runs a full inner `quad`
 * (whose limits may depend on the outer variables). Tessero reuses the verified adaptive {@see Quad::quad}
 * engine as the one-dimensional primitive, so the returned value matches scipy to the requested tolerance
 * (scipy's defaults epsabs = epsrel = 1.49e-8). Pure PHP: calls the user's callable, no kernel.
 */
final class NestedQuad
{
    /**
     * Double integral of $func(y, x) with the inner variable y over [$gfun($x), $hfun($x)] and the outer
     * variable x over [$a, $b] (scipy.integrate.dblquad signature and argument order).
     *
     * @param callable(float, float): float $func  integrand func(y, x)
     * @param callable(float): float        $gfun  inner lower limit as a function of x (or pass a constant via a closure)
     * @param callable(float): float        $hfun  inner upper limit as a function of x
     * @return array{0: float, 1: float}  [value, abserr]
     */
    public static function dblquad(callable $func, float $a, float $b, callable $gfun, callable $hfun, float $epsabs = 1.49e-8, float $epsrel = 1.49e-8): array
    {
        $outer = static function (float $x) use ($func, $gfun, $hfun, $epsabs, $epsrel): float {
            $inner = static fn (float $y): float => (float) $func($y, $x);
            [$v] = Quad::quad($inner, (float) $gfun($x), (float) $hfun($x), $epsabs, $epsrel);
            return $v;
        };
        return Quad::quad($outer, $a, $b, $epsabs, $epsrel);
    }

    /**
     * Triple integral of $func(z, y, x): z over [$qfun($x,$y), $rfun($x,$y)], y over [$gfun($x), $hfun($x)],
     * x over [$a, $b] (scipy.integrate.tplquad signature and argument order).
     *
     * @param callable(float, float, float): float $func  integrand func(z, y, x)
     * @param callable(float): float                $gfun  middle lower limit g(x)
     * @param callable(float): float                $hfun  middle upper limit h(x)
     * @param callable(float, float): float         $qfun  inner lower limit q(x, y)
     * @param callable(float, float): float         $rfun  inner upper limit r(x, y)
     * @return array{0: float, 1: float}  [value, abserr]
     */
    public static function tplquad(callable $func, float $a, float $b, callable $gfun, callable $hfun, callable $qfun, callable $rfun, float $epsabs = 1.49e-8, float $epsrel = 1.49e-8): array
    {
        $inner2d = static function (float $y, float $x) use ($func, $qfun, $rfun, $epsabs, $epsrel): float {
            $innermost = static fn (float $z): float => (float) $func($z, $y, $x);
            [$v] = Quad::quad($innermost, (float) $qfun($x, $y), (float) $rfun($x, $y), $epsabs, $epsrel);
            return $v;
        };
        return self::dblquad($inner2d, $a, $b, $gfun, $hfun, $epsabs, $epsrel);
    }

    /**
     * Integration over multiple variables (scipy.integrate.nquad) with fixed rectangular limits.
     *
     * $func takes the integration variables in the order x0, x1, ..., x_{m-1}; $ranges is a list of
     * [low, high] pairs, one per variable, with $ranges[0] the OUTERMOST (scipy integrates the LAST range
     * first/innermost — here the ranges are given outer-to-inner to match the callable's positional order).
     * Recursively reduces to adaptive one-dimensional `quad` on each axis.
     *
     * @param callable(float...): float      $func    integrand of m variables
     * @param list<array{0: float, 1: float}> $ranges  m fixed [low, high] pairs, outermost first
     * @return array{0: float, 1: float}  [value, abserr]
     */
    public static function nquad(callable $func, array $ranges, float $epsabs = 1.49e-8, float $epsrel = 1.49e-8): array
    {
        $ranges = array_values($ranges);
        $recurse = static function (array $fixed) use (&$recurse, $func, $ranges, $epsabs, $epsrel): array {
            $depth = count($fixed);
            [$lo, $hi] = $ranges[$depth];
            $isLast = $depth === count($ranges) - 1;
            $g = static function (float $xi) use ($recurse, $func, $fixed, $isLast): float {
                $args = $fixed;
                $args[] = $xi;
                if ($isLast) {
                    return (float) $func(...$args);
                }
                [$v] = $recurse($args);
                return $v;
            };
            return Quad::quad($g, (float) $lo, (float) $hi, $epsabs, $epsrel);
        };
        return $recurse([]);
    }
}
