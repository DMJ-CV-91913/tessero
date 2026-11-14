<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Exceptions\ConvergenceError;
use Tessero\Exceptions\TesseroException;

/**
 * Non-linear least squares with MINPACK's Levenberg-Marquardt code as SciPy runs it (kernel linalg.lmdif): solve as
 * scipy.optimize.leastsq, curveFit as curve_fit, leastSquares as least_squares with method 'lm'.
 */
final class LeastSquares
{
    /**
     * scipy.optimize.leastsq: minimise sum(residuals(p)^2) with MINPACK's lmdif, or lmder with $jac, with leastsq's
     * defaults (maxfev 200 (n + 1), with $jac 100 (n + 1)), so x, the residuals, nfev and the status follow SciPy's.
     * status is leastsq's ier (1 .. 4: a solution found) and message its mesg; cost is half the sum of squares, cov_x
     * leastsq's cov_x (null when R is singular), jac the Jacobian at x ($jac's, else 2-point differences, which are
     * not counted in nfev).
     *
     * @param callable(list<float>): list<float> $residuals
     * @param list<float> $p0
     * @param (callable(list<float>): list<list<float>>)|null $jac  the Jacobian, one row per residual
     * @return array{x: list<float>, cost: float, fun: list<float>, jac: list<list<float>>, nfev: int, njev: int, status: int, message: string, success: bool, cov_x: list<list<float>>|null}
     */
    public static function solve(callable $residuals, array $p0, ?callable $jac = null, float $ftol = 1.49012e-8, float $xtol = 1.49012e-8, float $gtol = 0.0, ?int $maxfev = null): array
    {
        $p0 = array_map('floatval', array_values($p0));
        $maxfev ??= ($jac === null ? 200 : 100) * (count($p0) + 1);
        [$x, $fvec, $info, $nfev, $cov, $sumsq, $njev] = self::minpack($residuals, $p0, $jac, $jac === null ? 0 : 1, $ftol, $xtol, $gtol, $maxfev);
        $J = $jac !== null ? self::rows($jac($x)) : self::twoPoint($residuals, $x, $fvec);

        return ['x' => $x, 'cost' => 0.5 * $sumsq, 'fun' => $fvec, 'jac' => $J, 'nfev' => $nfev, 'njev' => $njev, 'status' => $info,
            'message' => self::lmdifMessage($info, $ftol, $xtol, $gtol, $maxfev), 'success' => $info >= 1 && $info <= 4, 'cov_x' => $cov];
    }

    /**
     * MINPACK's Levenberg-Marquardt through linalg.lmdif, evaluating the residuals (and with jac 1 the Jacobian) at
     * the points it asks for: [x, fvec, info, nfev, cov_x or null, sum(fvec^2), njev].
     *
     * @param list<float> $p0
     * @return array{0: list<float>, 1: list<float>, 2: int, 3: int, 4: list<list<float>>|null, 5: float, 6: int}
     */
    private static function minpack(callable $residuals, array $p0, ?callable $jac, int $mode, float $ftol, float $xtol, float $gtol, int $maxfev): array
    {
        $lin = extension_loaded('tessero') && class_exists(\Tessero\Ext\Linalg::class) ? \Tessero\Ext\Linalg::class : \Tessero\Linalg::class;
        $n = count($p0);
        [$state, $task, $pts] = array_values($lin::lmdif(null, null, false, $p0, null, null, $ftol, $xtol, $gtol, $maxfev, 0.0, 100.0, $mode));
        while ($task !== 0) {
            $P = $pts->toArray();
            if ($task < 0) {
                $F = self::rows($jac($P[0]));
            } else {
                $F = [];
                foreach ($P as $p) {
                    $r = $residuals($p);
                    $F[] = is_object($r) ? $r->toList() : array_map('floatval', array_values($r));
                }
                if (count($F[0]) < $n) {
                    throw new TesseroException('least squares: ' . count($F[0]) . " residuals for {$n} parameters (need m >= n).");
                }
            }
            [$state, $task, $pts] = array_values($lin::lmdif($state, $F, false));
        }
        [$x, $fvec, $info, $nfev, $cov, $sumsq, $njev] = array_values($lin::lmdifResult($state));
        $cov = $cov->toArray();
        foreach ($cov as $row) {
            foreach ($row as $v) {
                if (is_nan($v)) {
                    $cov = null;
                    break 2;
                }
            }
        }

        return [$x->toList(), $fvec->toList(), $info, $nfev, $cov, $sumsq, $njev];
    }

    /**
     * The Jacobian at x by SciPy's 2-point differences (approx_derivative): h = EPS**0.5 * sign(x_j) * max(1, |x_j|),
     * column j (f(x + h e_j) - f0) / ((x_j + h) - x_j); one row per residual.
     *
     * @param list<float> $x
     * @param list<float> $f0
     * @return list<list<float>>
     */
    private static function twoPoint(callable $residuals, array $x, array $f0): array
    {
        $rstep = sqrt(PHP_FLOAT_EPSILON);
        $J = array_fill(0, count($f0), []);
        foreach ($x as $j => $xj) {
            $x1 = $x;
            $x1[$j] = $xj + ($rstep * ($xj >= 0.0 ? 1.0 : -1.0)) * max(1.0, abs($xj));
            $dx = $x1[$j] - $xj;
            $r = $residuals($x1);
            $f1 = is_object($r) ? $r->toList() : array_values($r);
            foreach ($f0 as $i => $v) {
                $J[$i][] = ((float) $f1[$i] - $v) / $dx;
            }
        }

        return $J;
    }

    /** @return list<list<float>> a Jacobian as rows, from a nested list or an array object */
    private static function rows(mixed $J): array
    {
        return is_object($J) ? $J->toArray() : array_values($J);
    }

    /**
     * scipy.optimize.curve_fit (method 'lm', SciPy's choice without bounds): fit f(x, ...params) to data with
     * MINPACK's lmdif as SciPy runs it (linalg.lmdif: SciPy's own code, leastsq's defaults), so the parameters, the
     * number of evaluations and the covariance follow SciPy's. Returns [popt, pcov], with $fullOutput
     * [popt, pcov, infodict (nfev, fvec), mesg, ier] as SciPy's full_output.
     *
     * The model is called the way xdata comes in. A PHP list: once per point, f(float $x, float ...$p) returning a
     * number. An NDArray (of either backend): on all of xdata, f(NDArray $x, float ...$p) returning one value per
     * point (an NDArray of that backend or a list), as SciPy calls it, the data staying in the kernel. For the n
     * evaluations of each forward-difference Jacobian such a model is first offered them in one call, x as a column
     * (m, 1) and each parameter as a row (1, n) of the n evaluation points; it keeps that call when it returns an
     * (m, n) array with the same bits as the n separate calls (compared once), and is called n times otherwise.
     *
     * @param callable $f                         the model, per point or on the whole of xdata (see above)
     * @param list<float|int>|object $xdata       a list, or an NDArray of either backend
     * @param list<float|int>|object $ydata
     * @param list<float|int> $p0
     * @param list<float|int>|object|null $sigma  one value per point, or a single value
     * @return array{0: list<float>, 1: list<list<float>>}|array{0: list<float>, 1: list<list<float>>, 2: array{nfev: int, fvec: list<float>}, 3: string, 4: int}
     */
    public static function curveFit(callable $f, array|object $xdata, array|object $ydata, array $p0, array|object|null $sigma = null, bool $absoluteSigma = false, bool $fullOutput = false): array
    {
        $p0 = array_map('floatval', array_values($p0));
        $n = count($p0);
        $vec = is_object($xdata);
        $ext = class_exists(\Tessero\Ext\NDArray::class, false)
            && ($vec ? $xdata instanceof \Tessero\Ext\NDArray : extension_loaded('tessero') && class_exists(\Tessero\Ext\Linalg::class));
        $nd = $ext ? \Tessero\Ext\NDArray::class : \Tessero\NDArray::class;
        $lin = $ext ? \Tessero\Ext\Linalg::class : \Tessero\Linalg::class;
        // arrays go to the kernel as they are when they belong to the backend used, else as lists
        $arg = static fn (mixed $v): mixed => is_object($v) ? ($v instanceof $nd ? $v : $v->toList()) : ($v === null ? null : array_map('floatval', array_values($v)));
        $y = $arg($ydata);
        $m = is_object($y) ? $y->size() : count($y);
        if ($n > $m) {
            throw new TesseroException("curveFit: the number of parameters ({$n}) must not exceed the number of data points ({$m}).");
        }
        if ($vec) {
            $x = $xdata instanceof $nd ? $xdata->astype('float64') : $nd::array($xdata->toList(), 'float64');
        } else {
            $x = array_values($xdata);
            if (count($x) !== $m) {
                throw new TesseroException('curveFit: xdata and ydata differ in length.');
            }
        }
        $maxfev = 200 * ($n + 1);
        [$state, $task, $pts] = array_values($lin::lmdif(null, null, false, $p0, $y, $arg($sigma), 1.49012e-8, 1.49012e-8, 0.0, $maxfev, 0.0, 100.0));
        $together = null;                                // the Jacobian's evaluations in one call: null until tried
        while ($task > 0) {
            $P = $pts->toArray();
            [$F, $fcols] = $vec ? self::cfOnArrays($f, $x, $P, $m, $nd, $together) : [self::cfPerPoint($f, $x, $P), false];
            [$state, $task, $pts] = array_values($lin::lmdif($state, $F, $fcols));
        }
        [$px, $fvec, $info, $nfev, $cov, $sumsq] = array_values($lin::lmdifResult($state));
        $popt = $px->toList();
        $mesg = self::lmdifMessage($info, 1.49012e-8, 1.49012e-8, 0.0, $maxfev);
        if ($info < 1 || $info > 4) {
            throw new ConvergenceError('curveFit: optimal parameters not found: ' . $mesg);
        }
        $pcov = $cov->toArray();
        $undefined = false;
        foreach ($pcov as $row) {
            foreach ($row as $v) {
                $undefined = $undefined || is_nan($v);
            }
        }
        if ($undefined || (! $absoluteSigma && $m <= $n)) {
            $pcov = array_fill(0, $n, array_fill(0, $n, INF));
        } elseif (! $absoluteSigma) {
            $s2 = $sumsq / ($m - $n);
            foreach ($pcov as $i => $row) {
                foreach ($row as $j => $v) {
                    $pcov[$i][$j] = $v * $s2;
                }
            }
        }
        if (! $fullOutput) {
            return [$popt, $pcov];
        }

        return [$popt, $pcov, ['nfev' => $nfev, 'fvec' => $fvec->toList()], $mesg, $info];
    }

    /**
     * curveFit's per-point model at each point lmdif asks for: one row of values per point. Up to four parameters
     * are passed without the argument spread (fewer PHP operations per point, the same values).
     *
     * @param list<float|int> $xdata
     * @param list<list<float>> $P
     * @return list<list<float>>
     */
    private static function cfPerPoint(callable $f, array $xdata, array $P): array
    {
        $F = [];
        foreach ($P as $p) {
            $out = [];
            switch (count($p)) {
                case 1:
                    [$a] = $p;
                    foreach ($xdata as $x) {
                        $out[] = (float) $f($x, $a);
                    }
                    break;
                case 2:
                    [$a, $b] = $p;
                    foreach ($xdata as $x) {
                        $out[] = (float) $f($x, $a, $b);
                    }
                    break;
                case 3:
                    [$a, $b, $c] = $p;
                    foreach ($xdata as $x) {
                        $out[] = (float) $f($x, $a, $b, $c);
                    }
                    break;
                case 4:
                    [$a, $b, $c, $d] = $p;
                    foreach ($xdata as $x) {
                        $out[] = (float) $f($x, $a, $b, $c, $d);
                    }
                    break;
                default:
                    foreach ($xdata as $x) {
                        $out[] = (float) $f($x, ...$p);
                    }
            }
            $F[] = $out;
        }

        return $F;
    }

    /**
     * curveFit's whole-array model at the points lmdif asks for: [values, fcols], the values one row per point
     * (fcols false), or the (m, k) result of the one call for k points (fcols true). $together records whether
     * that one call is used: tried on the first Jacobian and kept only when it has the bits of the k calls.
     *
     * @param list<list<float>> $P
     * @return array{0: mixed, 1: bool}
     */
    private static function cfOnArrays(callable $f, object $x, array $P, int $m, string $nd, ?bool &$together): array
    {
        $k = count($P);
        $values = static function (mixed $v) use ($nd, $m): mixed {
            if ($v instanceof $nd) {
                return $v;
            }
            if (is_object($v) && method_exists($v, 'toList')) {
                return $v->toList();
            }
            if (! is_array($v) || count($v) !== $m) {
                throw new TesseroException("curveFit: the model must return {$m} values (one per point) for xdata as an array.");
            }

            return array_map('floatval', array_values($v));
        };
        if ($k === 1) {
            return [$values($f($x, ...$P[0])), false];
        }
        $B = null;
        if ($together !== false && $x->ndim() === 1) {
            try {
                $cols = [];
                foreach (array_keys($P[0]) as $i) {
                    $cols[] = $nd::array([array_column($P, $i)], 'float64');
                }
                $B = $f($x->reshape($m, 1), ...$cols);
            } catch (\Throwable) {
                $B = null;
            }
            if (! $B instanceof $nd || $B->shape() !== [$m, $k]) {
                $B = null;
            }
        }
        if ($B !== null && $together === true) {
            return [$B, true];
        }
        $rows = [];
        $raw = '';
        foreach ($P as $p) {
            $v = $values($f($x, ...$p));
            if ($together === null && $B !== null) {
                $raw .= is_object($v) ? $v->astype('float64')->toBytes() : pack('e*', ...$v);
            }
            $rows[] = is_object($v) ? $v->toList() : $v;
        }
        if ($together === null) {
            // the one call's columns against the k calls, as raw little-endian doubles (C order of its transpose)
            $together = $B !== null && $B->astype('float64')->transpose()->toBytes() === $raw;
        }

        return [$rows, false];
    }

    /** leastsq's message for MINPACK's info */
    private static function lmdifMessage(int $info, float $ftol, float $xtol, float $gtol, int $maxfev): string
    {
        return match ($info) {
            0 => 'Improper input parameters.',
            1 => sprintf("Both actual and predicted relative reductions in the sum of squares\n  are at most %f", $ftol),
            2 => sprintf('The relative error between two consecutive iterates is at most %f', $xtol),
            3 => sprintf("Both actual and predicted relative reductions in the sum of squares\n  are at most %f and the relative error between two consecutive iterates is at \n  most %f", $ftol, $xtol),
            4 => sprintf("The cosine of the angle between func(x) and any column of the\n  Jacobian is at most %f in absolute value", $gtol),
            5 => "Number of calls to function has reached maxfev = {$maxfev}.",
            6 => sprintf("ftol=%f is too small, no further reduction in the sum of squares\n  is possible.", $ftol),
            7 => sprintf("xtol=%f is too small, no further improvement in the approximate\n  solution is possible.", $xtol),
            8 => sprintf("gtol=%f is too small, func(x) is orthogonal to the columns of\n  the Jacobian to machine precision.", $gtol),
            default => "Unknown error (info {$info}).",
        };
    }

    /**
     * scipy.optimize.least_squares with method 'lm' (unbounded): minimise 0.5 ||fun(x)||^2 from x0 with MINPACK's lmder
     * as SciPy 1.17 runs it, the Jacobian $jac's or approx_derivative's 2-point differences, max_nfev 100 n by default;
     * x, nfev, njev ($jac only), status and message follow SciPy's. fun is the cost (half the sum of squares,
     * summed pairwise, where SciPy uses its BLAS dot) and jac the Jacobian at x (one row per residual), as SciPy's jac.
     * SciPy's default method 'trf' takes other steps to the same minimum. Bounds and the trf/dogbox trust-region
     * methods are not supported.
     *
     * @param callable(list<float>): list<float> $fun
     * @param list<float|int> $x0
     * @param (callable(list<float>): list<list<float>>)|null $jac
     */
    public static function leastSquares(callable $fun, array $x0, ?callable $jac = null, float $ftol = 1e-8, float $xtol = 1e-8, float $gtol = 1e-8, ?int $maxNfev = null): OptimizeResult
    {
        $x0 = array_map('floatval', array_values($x0));
        if ($ftol < PHP_FLOAT_EPSILON || $xtol < PHP_FLOAT_EPSILON || $gtol < PHP_FLOAT_EPSILON) {
            throw new TesseroException("leastSquares: all tolerances must be higher than machine epsilon (2.22e-16) for method 'lm'.");
        }
        if ($maxNfev !== null && $maxNfev <= 0) {
            throw new TesseroException('leastSquares: maxNfev must be null or a positive integer.');
        }
        $maxNfev ??= 100 * count($x0);
        [$x, $fvec, $info, $nfev, , $sumsq, $njev] = self::minpack($fun, $x0, $jac, $jac === null ? 2 : 1, $ftol, $xtol, $gtol, $maxNfev);
        $J = $jac !== null ? self::rows($jac($x)) : self::twoPoint($fun, $x, $fvec);
        // MINPACK's info as least_squares' status (6 .. 8 cannot occur with every tolerance above machine epsilon)
        $status = [0 => -1, 1 => 2, 2 => 3, 3 => 4, 4 => 1, 5 => 0][$info];
        $message = [
            -1 => 'Improper input parameters status returned from `leastsq`',
            0 => 'The maximum number of function evaluations is exceeded.',
            1 => '`gtol` termination condition is satisfied.',
            2 => '`ftol` termination condition is satisfied.',
            3 => '`xtol` termination condition is satisfied.',
            4 => 'Both `ftol` and `xtol` termination conditions are satisfied.',
        ][$status];

        return new OptimizeResult(x: $x, fun: 0.5 * $sumsq, success: $status > 0, status: $status, message: $message, nit: 0, nfev: $nfev,
            njev: $jac !== null ? $njev : 0, jac: $J, method: 'lm');
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
    /**
     * Isotonic (monotonic) regression by the Pool Adjacent Violators Algorithm
     * (scipy.optimize.isotonic_regression): the weighted least-squares best monotonic fit to $y.
     *
     * @param list<int|float>      $y
     * @param list<int|float>|null $weights
     * @return list<float>
     */
    public static function isotonicRegression(array $y, ?array $weights = null, bool $increasing = true): array
    {
        $y = array_map('floatval', array_values($y));
        $n = count($y);
        if ($n === 0) {
            return [];
        }
        $w = $weights === null ? array_fill(0, $n, 1.0) : array_map('floatval', array_values($weights));
        if (! $increasing) {
            $y = array_reverse($y);
            $w = array_reverse($w);
        }
        $vals = [];
        $wts = [];
        $sizes = [];
        for ($i = 0; $i < $n; $i++) {
            $v = $y[$i];
            $ww = $w[$i];
            $sz = 1;
            while (($c = count($vals)) > 0 && $vals[$c - 1] >= $v) {
                $pv = array_pop($vals);
                $pw = array_pop($wts);
                $ps = array_pop($sizes);
                $v = ($pv * $pw + $v * $ww) / ($pw + $ww);
                $ww = $pw + $ww;
                $sz = $ps + $sz;
            }
            $vals[] = $v;
            $wts[] = $ww;
            $sizes[] = $sz;
        }
        $out = [];
        foreach ($vals as $k => $v) {
            for ($j = 0; $j < $sizes[$k]; $j++) {
                $out[] = $v;
            }
        }

        return $increasing ? $out : array_reverse($out);
    }

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
