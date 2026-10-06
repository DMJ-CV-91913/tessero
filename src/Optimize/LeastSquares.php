<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Exceptions\ConvergenceError;
use Tessero\Exceptions\TesseroException;

/**
 * Non-linear least squares by Levenberg-Marquardt (scipy.optimize.least_squares
 * method='lm' / curve_fit) with Nielsen's damping update and forward-difference
 * Jacobians.
 */
final class LeastSquares
{
    /**
     * Minimise 0.5 * sum(residuals(p)^2).
     *
     * @param callable(list<float>): list<float> $residuals
     * @param list<float> $p0
     * @param (callable(list<float>): list<list<float>>)|null $jac
     * @return array{x: list<float>, cost: float, fun: list<float>, jac: list<list<float>>, nfev: int, status: int, message: string, success: bool}
     */
    public static function solve(callable $residuals, array $p0, ?callable $jac = null, float $ftol = 1e-8, float $xtol = 1e-8, float $gtol = 1e-8, ?int $maxfev = null): array
    {
        $p = array_map('floatval', array_values($p0));
        $n = count($p);
        $maxfev ??= 100 * ($n + 1);
        $nfev = 0;
        $R = static function (array $x) use ($residuals, &$nfev): array {
            $nfev++;

            return array_map('floatval', array_values($residuals($x)));
        };
        $r = $R($p);
        $m = count($r);
        if ($m < $n) {
            throw new TesseroException("least squares: {$m} residuals for {$n} parameters (need m >= n).");
        }
        $cost = 0.5 * Minimize::dot($r, $r);
        $J = $jac !== null ? $jac($p) : Root::fdJacobian($R, $p, $r);
        [$A, $g] = self::normal($J, $r);
        $mu = 1e-3 * max(array_map(static fn (int $i): float => $A[$i][$i], range(0, $n - 1)) ?: [1.0]);
        $nu = 2.0;
        $status = 0;
        $message = 'maxfev reached';
        while ($nfev < $maxfev) {
            if (max(array_map('abs', $g)) <= $gtol) {
                $status = 1;
                $message = 'gtol: the gradient is small enough';
                break;
            }
            $Aug = $A;
            for ($i = 0; $i < $n; $i++) {
                $Aug[$i][$i] = $A[$i][$i] + $mu * max($A[$i][$i], 1e-12);
            }
            try {
                $step = Dense::solve($Aug, array_map(static fn (float $v): float => -$v, $g));
            } catch (\Throwable) {
                $mu *= $nu;
                $nu *= 2;
                continue;
            }
            $stepNorm = sqrt(Minimize::dot($step, $step));
            $pNorm = sqrt(Minimize::dot($p, $p));
            if ($stepNorm <= $xtol * ($pNorm + $xtol)) {
                $status = 2;
                $message = 'xtol: the parameter change is small enough';
                break;
            }
            $pNew = [];
            foreach ($p as $i => $v) {
                $pNew[] = $v + $step[$i];
            }
            $rNew = $R($pNew);
            $costNew = 0.5 * Minimize::dot($rNew, $rNew);
            // gain ratio: actual / predicted reduction
            $pred = 0.0;
            for ($i = 0; $i < $n; $i++) {
                $pred += 0.5 * $step[$i] * ($mu * max($A[$i][$i], 1e-12) * $step[$i] - $g[$i]);
            }
            $rho = $pred > 0 ? ($cost - $costNew) / $pred : -1.0;
            if ($rho > 0 && is_finite($costNew)) {
                $reduction = ($cost - $costNew) / max($cost, PHP_FLOAT_MIN);
                $p = $pNew;
                $r = $rNew;
                $cost = $costNew;
                $J = $jac !== null ? $jac($p) : Root::fdJacobian($R, $p, $r);
                [$A, $g] = self::normal($J, $r);
                $mu *= max(1 / 3, 1 - (2 * $rho - 1) ** 3);
                $nu = 2.0;
                if ($reduction <= $ftol) {
                    $status = 3;
                    $message = 'ftol: the relative reduction of the cost is small enough';
                    break;
                }
            } else {
                $mu *= $nu;
                $nu *= 2;
            }
        }

        return ['x' => $p, 'cost' => $cost, 'fun' => $r, 'jac' => $J, 'nfev' => $nfev, 'status' => $status, 'message' => $message, 'success' => $status > 0];
    }

    /**
     * scipy.optimize.curve_fit: fit f(x, ...params) to data. Returns [popt, pcov].
     *
     * @param callable(float, float ...): float $f       model evaluated at one x value
     * @param list<float> $xdata
     * @param list<float> $ydata
     * @param list<float> $p0
     * @param list<float>|null $sigma
     * @return array{0: list<float>, 1: list<list<float>>}
     */
    public static function curveFit(callable $f, array $xdata, array $ydata, array $p0, ?array $sigma = null, bool $absoluteSigma = false): array
    {
        $xdata = array_values($xdata);
        $ydata = array_values($ydata);
        if (count($xdata) !== count($ydata)) {
            throw new TesseroException('curveFit: xdata and ydata differ in length.');
        }
        $w = $sigma === null ? null : array_map(static fn ($s): float => 1.0 / (float) $s, array_values($sigma));
        $res = static function (array $p) use ($f, $xdata, $ydata, $w): array {
            $out = [];
            foreach ($xdata as $i => $x) {
                $d = (float) $f($x, ...$p) - $ydata[$i];
                $out[] = $w === null ? $d : $d * $w[$i];
            }

            return $out;
        };
        $fit = self::solve($res, $p0);
        if (! $fit['success']) {
            throw new ConvergenceError('curveFit: optimal parameters not found (' . $fit['message'] . ').');
        }
        $n = count($p0);
        $m = count($ydata);
        [$A] = self::normal($fit['jac'], $fit['fun']);
        try {
            $cov = Dense::inverse($A);
        } catch (\Throwable) {
            $cov = array_fill(0, $n, array_fill(0, $n, INF));
        }
        if (! $absoluteSigma) {
            $s2 = $m > $n ? 2 * $fit['cost'] / ($m - $n) : INF;
            foreach ($cov as $i => $row) {
                foreach ($row as $j => $v) {
                    $cov[$i][$j] = $v * $s2;
                }
            }
        }

        return [$fit['x'], $cov];
    }

    /**
     * scipy.optimize.least_squares (unbounded): minimise 0.5 ||fun(x)||^2 from x0. For an unbounded problem the
     * minimiser is that of the Levenberg-Marquardt solve (self::solve), so this wraps it in an OptimizeResult.
     * Bounds / the trf/dogbox trust-region variants are not yet supported.
     *
     * @param callable(list<float>): list<float> $fun
     * @param list<float|int> $x0
     */
    public static function leastSquares(callable $fun, array $x0): OptimizeResult
    {
        $r = self::solve($fun, array_map('floatval', array_values($x0)));
        $x = $r['x'];
        $res = $fun($x);
        $cost = 0.0;
        foreach ($res as $ri) {
            $cost += (float) $ri * (float) $ri;
        }

        return new OptimizeResult(x: $x, fun: 0.5 * $cost, success: true, status: 1, message: '`ftol` termination condition is satisfied.', nit: 0, nfev: 0);
    }

    /**
     * scipy.optimize.lsq_linear (unbounded): the linear least-squares solution of min 0.5||A x - b||^2, returned
     * as an OptimizeResult. The unbounded solution is the ordinary least-squares solution (normal equations);
     * finite bounds (the trf/bvls variants) are not yet supported.
     *
     * @param list<list<float>> $A
     * @param list<float> $b
     * @param array{0: mixed, 1: mixed}|null $bounds
     */
    public static function lsqLinear(array $A, array $b, ?array $bounds = null): OptimizeResult
    {
        if ($bounds !== null) {
            foreach ($bounds as $bnd) {
                foreach ((is_array($bnd) ? $bnd : [$bnd]) as $val) {
                    if (is_finite((float) $val)) {
                        throw new TesseroException('lsq_linear: finite bounds are not yet supported (unbounded only).');
                    }
                }
            }
        }
        [$ata, $atb] = self::normal($A, $b);
        $x = self::solveSym($ata, $atb);
        $m = count($A);
        $n = count($x);
        $cost = 0.0;
        for ($i = 0; $i < $m; $i++) {
            $s = -$b[$i];
            for ($j = 0; $j < $n; $j++) {
                $s += $A[$i][$j] * $x[$j];
            }
            $cost += $s * $s;
        }

        return new OptimizeResult(x: $x, fun: 0.5 * $cost, success: true, status: 1, message: 'The unconstrained solution is optimal.', nit: 0, nfev: 0);
    }

    /**
     * scipy.optimize.nnls: the non-negative least-squares solution of min ||A x - b|| subject to x >= 0, by the
     * classical Lawson-Hanson active-set method. The solution is unique, so it matches scipy. Returns [x, rnorm].
     *
     * @param list<list<float>> $A
     * @param list<float> $b
     * @return array{0: list<float>, 1: float}
     */
    public static function nnls(array $A, array $b, ?int $maxiter = null): array
    {
        $m = count($A);
        $n = count($A[0] ?? []);
        $maxiter ??= 3 * $n;
        [$ata, $atb] = self::normal($A, $b);          // A^T A (n x n), A^T b (n)
        $eps = 2.220446049250313e-16;
        $amax = 0.0;
        foreach ($A as $row) {
            foreach ($row as $v) {
                $amax = max($amax, abs($v));
            }
        }
        $tol = 10.0 * $eps * max($m, $n) * ($amax > 0.0 ? $amax : 1.0);
        $x = array_fill(0, $n, 0.0);
        $passive = array_fill(0, $n, false);
        $iter = 0;
        while (true) {
            $w = $atb;                                 // w = A^T(b - A x) = A^T b - (A^T A) x
            for ($i = 0; $i < $n; $i++) {
                $s = 0.0;
                for ($j = 0; $j < $n; $j++) {
                    $s += $ata[$i][$j] * $x[$j];
                }
                $w[$i] -= $s;
            }
            $jstar = -1;
            $wmax = $tol;
            for ($j = 0; $j < $n; $j++) {
                if (!$passive[$j] && $w[$j] > $wmax) {
                    $wmax = $w[$j];
                    $jstar = $j;
                }
            }
            if ($jstar < 0) {
                break;                                 // KKT conditions satisfied
            }
            $passive[$jstar] = true;
            while (true) {
                if (++$iter > $maxiter) {
                    break 2;
                }
                $p = [];
                for ($j = 0; $j < $n; $j++) {
                    if ($passive[$j]) {
                        $p[] = $j;
                    }
                }
                $np = count($p);
                $sub = array_fill(0, $np, array_fill(0, $np, 0.0));
                $rhs = array_fill(0, $np, 0.0);
                for ($a = 0; $a < $np; $a++) {
                    $rhs[$a] = $atb[$p[$a]];
                    for ($c = 0; $c < $np; $c++) {
                        $sub[$a][$c] = $ata[$p[$a]][$p[$c]];
                    }
                }
                $zp = self::solveSym($sub, $rhs);
                $z = array_fill(0, $n, 0.0);
                $minz = INF;
                for ($a = 0; $a < $np; $a++) {
                    $z[$p[$a]] = $zp[$a];
                    $minz = min($minz, $zp[$a]);
                }
                if ($minz > 0.0) {
                    $x = $z;
                    break;
                }
                $alpha = INF;
                foreach ($p as $j) {
                    if ($z[$j] <= 0.0) {
                        $alpha = min($alpha, $x[$j] / ($x[$j] - $z[$j]));
                    }
                }
                for ($j = 0; $j < $n; $j++) {
                    $x[$j] += $alpha * ($z[$j] - $x[$j]);
                }
                for ($j = 0; $j < $n; $j++) {
                    if ($passive[$j] && abs($x[$j]) < $tol) {
                        $passive[$j] = false;
                        $x[$j] = 0.0;
                    }
                }
            }
        }
        $rn = 0.0;
        for ($i = 0; $i < $m; $i++) {
            $s = -$b[$i];
            for ($j = 0; $j < $n; $j++) {
                $s += $A[$i][$j] * $x[$j];
            }
            $rn += $s * $s;
        }

        return [$x, sqrt($rn)];
    }

    /** Solve the symmetric system M y = g by Gaussian elimination with partial pivoting. */
    private static function solveSym(array $M, array $g): array
    {
        $n = count($g);
        for ($k = 0; $k < $n; $k++) {
            $piv = $k;
            $mx = abs($M[$k][$k]);
            for ($i = $k + 1; $i < $n; $i++) {
                if (abs($M[$i][$k]) > $mx) {
                    $mx = abs($M[$i][$k]);
                    $piv = $i;
                }
            }
            if ($piv !== $k) {
                [$M[$k], $M[$piv]] = [$M[$piv], $M[$k]];
                [$g[$k], $g[$piv]] = [$g[$piv], $g[$k]];
            }
            $d = $M[$k][$k];
            if ($d == 0.0) {
                continue;
            }
            for ($i = $k + 1; $i < $n; $i++) {
                $f = $M[$i][$k] / $d;
                if ($f == 0.0) {
                    continue;
                }
                for ($j = $k; $j < $n; $j++) {
                    $M[$i][$j] -= $f * $M[$k][$j];
                }
                $g[$i] -= $f * $g[$k];
            }
        }
        $y = array_fill(0, $n, 0.0);
        for ($i = $n - 1; $i >= 0; $i--) {
            $s = $g[$i];
            for ($j = $i + 1; $j < $n; $j++) {
                $s -= $M[$i][$j] * $y[$j];
            }
            $y[$i] = ($M[$i][$i] != 0.0) ? $s / $M[$i][$i] : 0.0;
        }

        return $y;
    }

    /** @return array{0: list<list<float>>, 1: list<float>} J^T J and J^T r */
    private static function normal(array $J, array $r): array
    {
        $m = count($J);
        $n = count($J[0] ?? []);
        $A = array_fill(0, $n, array_fill(0, $n, 0.0));
        $g = array_fill(0, $n, 0.0);
        for ($k = 0; $k < $m; $k++) {
            $row = $J[$k];
            $rk = $r[$k];
            for ($i = 0; $i < $n; $i++) {
                $ji = $row[$i];
                if ($ji == 0.0) {
                    continue;
                }
                $g[$i] = $g[$i] + $ji * $rk;
                $Ai = $A[$i];
                for ($j = $i; $j < $n; $j++) {
                    $Ai[$j] = $Ai[$j] + $ji * $row[$j];
                }
                $A[$i] = $Ai;
            }
        }
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $i; $j++) {
                $A[$i][$j] = $A[$j][$i];
            }
        }

        return [$A, $g];
    }
}
