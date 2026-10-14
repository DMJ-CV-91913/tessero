<?php

declare(strict_types=1);

namespace Tessero\Differentiate;

/**
 * Adaptive finite-difference differentiation (scipy.differentiate: derivative / jacobian / hessian).
 *
 * Pure PHP: evaluates the user's callable, no kernel. Mirrors scipy.differentiate's default central-difference
 * scheme (step_direction=0). For an eighth-order (order=8) formula the stencil is
 *   x - h, x - h/c, x - h/c^2, x - h/c^3,  x,  x + h/c^3, x + h/c^2, x + h/c, x + h
 * where c = step_factor. The weights w solve the Vandermonde system A w = e_1 on the node offsets
 * s/c^(|i|-1) (i = -n..n), so that (f . w)/h approximates f'(x) to order `order`. Each iteration the step is
 * reduced by `c`; the derivative estimate df = (f . w)/h is recomputed and the error is estimated as the change
 * |df - df_last| from the previous iteration. Iteration stops when error < atol + rtol*|df| (converged, status 0),
 * when the error grows by more than 10x (floating-point cancellation floor, status -1), or at `maxiter`
 * (status 2 / not converged). Verified bit-for-bit against scipy.differentiate.derivative on the parity fixtures.
 *
 * jacobian/hessian apply the scalar scheme component-wise: the i-th partial derivative differentiates the
 * function along the i-th coordinate holding the others fixed; hessian differentiates each first partial again.
 */
final class Derivative
{
    /**
     * d/dx f(x) by adaptive central finite differences.
     *
     * @param callable(float): float $f
     * @return array{df: float, error: float, success: bool, status: int, nit: int, nfev: int}
     */
    public static function derivative(
        callable $f,
        float $x,
        ?float $atol = null,
        ?float $rtol = null,
        int $maxiter = 10,
        int $order = 8,
        float $initialStep = 0.5,
        float $stepFactor = 2.0
    ): array {
        $eps = 2.220446049250313e-16;
        if ($rtol === null) {
            $rtol = sqrt($eps);
        }
        if ($atol === null) {
            $atol = 0.0;
        }
        $n = intdiv($order + 1, 2);                   // half-order (terms each side)
        $w = self::centralWeights($n, $stepFactor);   // 2n+1 weights, w[n] = 0

        $h = $initialStep;
        $df = NAN;
        $dfLast = NAN;
        $error = NAN;
        $errorLast = NAN;
        $status = 2;                                  // _ECONVERGED=0, _EERRORINCREASE=-1, not-converged=2
        $nfev = 1;                                    // f(x) itself
        $nit = 0;

        for ($it = 0; $it < $maxiter; $it++) {
            // central abscissae: x - h/c^k (k=n-1..0) , x , x + h/c^k (k=0..n-1)
            $vals = [];
            for ($k = $n - 1; $k >= 0; $k--) {
                $vals[] = $f($x - $h / $stepFactor ** $k);
                $nfev++;
            }
            $vals[] = $f($x);                          // center (weight 0); kept for stencil symmetry
            for ($k = 0; $k < $n; $k++) {
                $vals[] = $f($x + $h / $stepFactor ** $k);
                $nfev++;
            }
            $dfLast = $df;
            $acc = 0.0;
            foreach ($w as $i => $wi) {
                $acc += $wi * $vals[$i];
            }
            $df = $acc / $h;
            $errorLast = $error;
            $error = abs($df - $dfLast);
            $nit = $it + 1;

            if ($error < $atol + $rtol * abs($df)) {
                $status = 0;
                break;
            }
            if ($it > 0 && $error > $errorLast * 10.0) {
                $status = -1;
                break;
            }
            $h /= $stepFactor;
        }

        return [
            'df' => $df,
            'error' => $error,
            'success' => $status === 0,
            'status' => $status,
            'nit' => $nit,
            'nfev' => $nfev,
        ];
    }

    /**
     * Jacobian of f: R^m -> R^k, evaluated at the point $x (length m). Returns a k-by-m array.
     *
     * @param callable(array<float>): array<float> $f
     * @param array<float> $x
     * @return array<int, array<int, float>>
     */
    public static function jacobian(callable $f, array $x, ?float $atol = null, ?float $rtol = null, int $maxiter = 10, int $order = 8, float $initialStep = 0.5, float $stepFactor = 2.0): array
    {
        $m = count($x);
        $f0 = $f($x);
        $k = count($f0);
        $cols = [];                                    // cols[j][out] = d f_out / d x_j
        for ($j = 0; $j < $m; $j++) {
            for ($out = 0; $out < $k; $out++) {
                $g = static function (float $t) use ($f, $x, $j, $out): float {
                    $xx = $x;
                    $xx[$j] = $t;
                    return $f($xx)[$out];
                };
                $cols[$j][$out] = self::derivative($g, $x[$j], $atol, $rtol, $maxiter, $order, $initialStep, $stepFactor)['df'];
            }
        }
        $jac = [];
        for ($out = 0; $out < $k; $out++) {
            for ($j = 0; $j < $m; $j++) {
                $jac[$out][$j] = $cols[$j][$out];
            }
        }
        return $jac;
    }

    /**
     * Hessian of a scalar function f: R^m -> R at the point $x. Returns an m-by-m array of second partials,
     * computed as the Jacobian of the numerical gradient.
     *
     * @param callable(array<float>): float $f
     * @param array<float> $x
     * @return array<int, array<int, float>>
     */
    public static function hessian(callable $f, array $x, ?float $atol = null, ?float $rtol = null, int $maxiter = 10, int $order = 8, float $initialStep = 0.5, float $stepFactor = 2.0): array
    {
        $grad = static function (array $xx) use ($f, $atol, $rtol, $maxiter, $order, $initialStep, $stepFactor): array {
            $m = count($xx);
            $g = [];
            for ($j = 0; $j < $m; $j++) {
                $one = static function (float $t) use ($f, $xx, $j): float {
                    $v = $xx;
                    $v[$j] = $t;
                    return $f($v);
                };
                $g[$j] = self::derivative($one, $xx[$j], $atol, $rtol, $maxiter, $order, $initialStep, $stepFactor)['df'];
            }
            return $g;
        };
        return self::jacobian($grad, $x, $atol, $rtol, $maxiter, $order, $initialStep, $stepFactor);
    }

    /**
     * Central finite-difference weights on the stencil offsets s/fac^(|i|-1), i = -n..n (w[n] forced to 0;
     * antisymmetry w[-i] = -w[i] enforced), matching scipy.differentiate._derivative_weights.
     *
     * @return array<int, float>
     */
    private static function centralWeights(int $n, float $fac): array
    {
        $size = 2 * $n + 1;
        $nodes = [];
        for ($idx = 0; $idx < $size; $idx++) {
            $i = $idx - $n;
            $s = $i <=> 0;                             // sign(i): -1, 0, +1
            $p = abs($i) - 1.0;                        // center -> p = -1, but s = 0 zeroes it
            $nodes[$idx] = $s === 0 ? 0.0 : $s / $fac ** $p;
        }
        // Vandermonde A[r][col] = nodes[col]^r ; solve A w = e_1 (second row picks out f').
        $A = [];
        for ($r = 0; $r < $size; $r++) {
            for ($col = 0; $col < $size; $col++) {
                $A[$r][$col] = $nodes[$col] ** $r;
            }
        }
        $b = array_fill(0, $size, 0.0);
        $b[1] = 1.0;
        $w = self::solve($A, $b);
        $w[$n] = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $w[$size - 1 - $i] = -$w[$i];
        }
        return $w;
    }

    /**
     * Dense Gaussian elimination with partial pivoting (small systems only).
     *
     * @param array<int, array<int, float>> $A
     * @param array<int, float> $b
     * @return array<int, float>
     */
    private static function solve(array $A, array $b): array
    {
        $n = count($b);
        for ($col = 0; $col < $n; $col++) {
            $piv = $col;
            for ($r = $col + 1; $r < $n; $r++) {
                if (abs($A[$r][$col]) > abs($A[$piv][$col])) {
                    $piv = $r;
                }
            }
            if ($piv !== $col) {
                [$A[$col], $A[$piv]] = [$A[$piv], $A[$col]];
                [$b[$col], $b[$piv]] = [$b[$piv], $b[$col]];
            }
            $d = $A[$col][$col];
            for ($r = $col + 1; $r < $n; $r++) {
                $factor = $A[$r][$col] / $d;
                if ($factor === 0.0) {
                    continue;
                }
                for ($c = $col; $c < $n; $c++) {
                    $A[$r][$c] -= $factor * $A[$col][$c];
                }
                $b[$r] -= $factor * $b[$col];
            }
        }
        $x = array_fill(0, $n, 0.0);
        for ($r = $n - 1; $r >= 0; $r--) {
            $s = $b[$r];
            for ($c = $r + 1; $c < $n; $c++) {
                $s -= $A[$r][$c] * $x[$c];
            }
            $x[$r] = $s / $A[$r][$r];
        }
        return $x;
    }
}
