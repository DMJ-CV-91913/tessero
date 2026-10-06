<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Exceptions\ConvergenceError;
use Tessero\Exceptions\TesseroException;

/** Scalar and vector root finding (scipy.optimize.brentq, newton, root). */
final class Root
{
    private const EPS = 2.220446049250313e-16;

    /**
     * Brent's method on a bracketing interval - a line-by-line port of
     * scipy/optimize/Zeros/brentq.c, so it returns the same root in the same
     * number of evaluations.
     *
     * @param callable(float): float $f
     * @return array{root: float, iterations: int, function_calls: int, converged: bool}
     */
    public static function brentq(callable $f, float $a, float $b, float $xtol = 2e-12, ?float $rtol = null, int $maxiter = 100, bool $fullOutput = false): float|array
    {
        $rtol ??= 4 * self::EPS;
        $xpre = $a;
        $xcur = $b;
        $xblk = 0.0;
        $fblk = 0.0;
        $spre = 0.0;
        $scur = 0.0;
        $fpre = (float) $f($xpre);
        $fcur = (float) $f($xcur);
        $calls = 2;
        $done = static fn (float $root, int $it, bool $ok) => $fullOutput ? ['root' => $root, 'iterations' => $it, 'function_calls' => $calls, 'converged' => $ok] : $root;
        if ($fpre == 0.0) {
            return $done($xpre, 0, true);
        }
        if ($fcur == 0.0) {
            return $done($xcur, 0, true);
        }
        if (($fpre < 0) === ($fcur < 0)) {
            throw new TesseroException('brentq: f(a) and f(b) must have different signs.');
        }
        for ($i = 0; $i < $maxiter; $i++) {
            if ($fpre != 0.0 && $fcur != 0.0 && (($fpre < 0) !== ($fcur < 0))) {
                $xblk = $xpre;
                $fblk = $fpre;
                $spre = $scur = $xcur - $xpre;
            }
            if (abs($fblk) < abs($fcur)) {
                $xpre = $xcur;
                $xcur = $xblk;
                $xblk = $xpre;
                $fpre = $fcur;
                $fcur = $fblk;
                $fblk = $fpre;
            }
            $delta = ($xtol + $rtol * abs($xcur)) / 2;
            $sbis = ($xblk - $xcur) / 2;
            if ($fcur == 0.0 || abs($sbis) < $delta) {
                return $fullOutput ? ['root' => $xcur, 'iterations' => $i + 1, 'function_calls' => $calls, 'converged' => true] : $xcur;
            }
            if (abs($spre) > $delta && abs($fcur) < abs($fpre)) {
                if ($xpre == $xblk) {
                    $stry = -$fcur * ($xcur - $xpre) / ($fcur - $fpre);
                } else {
                    $dpre = ($fpre - $fcur) / ($xpre - $xcur);
                    $dblk = ($fblk - $fcur) / ($xblk - $xcur);
                    $stry = -$fcur * ($fblk * $dblk - $fpre * $dpre) / ($dblk * $dpre * ($fblk - $fpre));
                }
                if (2 * abs($stry) < min(abs($spre), 3 * abs($sbis) - $delta)) {
                    $spre = $scur;
                    $scur = $stry;
                } else {
                    $spre = $sbis;
                    $scur = $sbis;
                }
            } else {
                $spre = $sbis;
                $scur = $sbis;
            }
            $xpre = $xcur;
            $fpre = $fcur;
            if (abs($scur) > $delta) {
                $xcur += $scur;
            } else {
                $xcur += ($sbis > 0 ? $delta : -$delta);
            }
            $fcur = (float) $f($xcur);
            $calls++;
        }
        if ($fullOutput) {
            return ['root' => $xcur, 'iterations' => $maxiter, 'function_calls' => $calls, 'converged' => false];
        }

        throw new ConvergenceError("brentq: failed to converge after {$maxiter} iterations.");
    }

    /**
     * Bisection on a bracketing interval - a port of scipy/optimize/Zeros/bisect.c, returning the same root.
     *
     * @param callable(float): float $f
     */
    public static function bisect(callable $f, float $a, float $b, float $xtol = 2e-12, ?float $rtol = null, int $maxiter = 100): float
    {
        $rtol ??= 4 * self::EPS;
        $fa = (float) $f($a);
        $fb = (float) $f($b);
        if ($fa == 0.0) {
            return $a;
        }
        if ($fb == 0.0) {
            return $b;
        }
        if (($fa < 0) === ($fb < 0)) {
            throw new TesseroException('bisect: f(a) and f(b) must have different signs.');
        }
        $dm = $b - $a;
        $xm = $a;
        for ($i = 0; $i < $maxiter; $i++) {
            $dm *= 0.5;
            $xm = $a + $dm;
            $fm = (float) $f($xm);
            if (($fm < 0) === ($fa < 0)) {
                $a = $xm;
            }
            if ($fm == 0.0 || abs($dm) < $xtol + $rtol * abs($xm)) {
                return $xm;
            }
        }

        throw new ConvergenceError("bisect: failed to converge after {$maxiter} iterations.");
    }

    /**
     * Brent's method with hyperbolic extrapolation - a port of scipy/optimize/Zeros/brenth.c (identical to brentq
     * except for the extrapolation step), returning the same root.
     *
     * @param callable(float): float $f
     */
    public static function brenth(callable $f, float $a, float $b, float $xtol = 2e-12, ?float $rtol = null, int $maxiter = 100): float
    {
        $rtol ??= 4 * self::EPS;
        $xpre = $a;
        $xcur = $b;
        $xblk = 0.0;
        $fblk = 0.0;
        $spre = 0.0;
        $scur = 0.0;
        $fpre = (float) $f($xpre);
        $fcur = (float) $f($xcur);
        if ($fpre == 0.0) {
            return $xpre;
        }
        if ($fcur == 0.0) {
            return $xcur;
        }
        if (($fpre < 0) === ($fcur < 0)) {
            throw new TesseroException('brenth: f(a) and f(b) must have different signs.');
        }
        for ($i = 0; $i < $maxiter; $i++) {
            if ($fpre != 0.0 && $fcur != 0.0 && (($fpre < 0) !== ($fcur < 0))) {
                $xblk = $xpre;
                $fblk = $fpre;
                $spre = $scur = $xcur - $xpre;
            }
            if (abs($fblk) < abs($fcur)) {
                $xpre = $xcur;
                $xcur = $xblk;
                $xblk = $xpre;
                $fpre = $fcur;
                $fcur = $fblk;
                $fblk = $fpre;
            }
            $delta = ($xtol + $rtol * abs($xcur)) / 2;
            $sbis = ($xblk - $xcur) / 2;
            if ($fcur == 0.0 || abs($sbis) < $delta) {
                return $xcur;
            }
            if (abs($spre) > $delta && abs($fcur) < abs($fpre)) {
                if ($xpre == $xblk) {
                    $stry = -$fcur * ($xcur - $xpre) / ($fcur - $fpre);
                } else {
                    $dpre = ($fpre - $fcur) / ($xpre - $xcur);
                    $dblk = ($fblk - $fcur) / ($xblk - $xcur);
                    $stry = -$fcur * ($fblk - $fpre) / ($fblk * $dpre - $fpre * $dblk);
                }
                if (2 * abs($stry) < min(abs($spre), 3 * abs($sbis) - $delta)) {
                    $spre = $scur;
                    $scur = $stry;
                } else {
                    $spre = $sbis;
                    $scur = $sbis;
                }
            } else {
                $spre = $sbis;
                $scur = $sbis;
            }
            $xpre = $xcur;
            $fpre = $fcur;
            if (abs($scur) > $delta) {
                $xcur += $scur;
            } else {
                $xcur += ($sbis > 0 ? $delta : -$delta);
            }
            $fcur = (float) $f($xcur);
        }

        throw new ConvergenceError("brenth: failed to converge after {$maxiter} iterations.");
    }

    /**
     * Ridder's method on a bracketing interval - a port of scipy/optimize/Zeros/ridder.c, returning the same root.
     *
     * @param callable(float): float $f
     */
    public static function ridder(callable $f, float $a, float $b, float $xtol = 2e-12, ?float $rtol = null, int $maxiter = 100): float
    {
        $rtol ??= 4 * self::EPS;
        $xa = $a;
        $xb = $b;
        $tol = $xtol + $rtol * min(abs($xa), abs($xb));
        $fa = (float) $f($xa);
        $fb = (float) $f($xb);
        if ($fa == 0.0) {
            return $xa;
        }
        if ($fb == 0.0) {
            return $xb;
        }
        if (($fa < 0.0) === ($fb < 0.0)) {
            throw new TesseroException('ridder: f(a) and f(b) must have different signs.');
        }
        $xn = $xb;
        for ($i = 0; $i < $maxiter; $i++) {
            $dm = 0.5 * ($xb - $xa);
            $xm = $xa + $dm;
            $fm = (float) $f($xm);
            if ($fm == 0.0) {
                return $xm;
            }
            $ratio = $fm / $fa;
            $dn = $dm * $ratio / sqrt($ratio * $ratio - $fb / $fa);
            $sign = ($dn < 0.0) ? -1.0 : 1.0;
            $xn = $xm + $sign * min(abs($dn), abs($dm) - 0.5 * $tol);
            $fn = (float) $f($xn);
            if (($fn < 0.0) !== ($fm < 0.0)) {
                $xa = $xn;
                $fa = $fn;
                $xb = $xm;
                $fb = $fm;
            } elseif (($fn < 0.0) !== ($fa < 0.0)) {
                $xb = $xn;
                $fb = $fn;
            } else {
                $xa = $xn;
                $fa = $fn;
            }
            $tol = $xtol + $rtol * $xn;
            if ($fn == 0.0 || abs($xb - $xa) < $tol) {
                return $xn;
            }
        }

        throw new ConvergenceError("ridder: failed to converge after {$maxiter} iterations.");
    }

    /**
     * Algorithm 748 of Alefeld, Potra and Shi (the k=1 form: a Newton-quadratic interpolation step, a
     * double-length secant step and a bisection safeguard), as scipy.optimize.toms748. Converges to the same
     * root as scipy.
     *
     * @param callable(float): float $f
     */
    public static function toms748(callable $f, float $a, float $b, float $xtol = 2e-12, ?float $rtol = null, int $maxiter = 100): float
    {
        $rtol ??= 4 * self::EPS;
        $fa = (float) $f($a);
        $fb = (float) $f($b);
        if ($fa == 0.0) {
            return $a;
        }
        if ($fb == 0.0) {
            return $b;
        }
        if (($fa < 0) === ($fb < 0)) {
            throw new TesseroException('toms748: f(a) and f(b) must have different signs.');
        }
        $d = 0.0;
        $fd = 0.0;
        $haveD = false;
        $rebracket = static function (float $c, float $fc, float &$a, float &$fa, float &$b, float &$fb, float &$d, float &$fd): void {
            if (($fa < 0) !== ($fc < 0)) {
                $d = $b;
                $fd = $fb;
                $b = $c;
                $fb = $fc;
            } else {
                $d = $a;
                $fd = $fa;
                $a = $c;
                $fa = $fc;
            }
        };
        for ($it = 0; $it < $maxiter; $it++) {
            $width = $b - $a;
            // step 1: Newton-quadratic interpolation (once a third point d is available), else secant
            if ($haveD && $b != $a && $d != $a && $d != $b) {
                $bb = ($fb - $fa) / ($b - $a);
                $aa = (($fd - $fb) / ($d - $b) - $bb) / ($d - $a);
                if ($aa == 0.0) {
                    $c = $a - $fa / $bb;
                } else {
                    $r = ((($aa < 0) === ($fa < 0)) && $aa != 0.0) ? $a : $b;
                    for ($s = 0; $s < 2; $s++) {
                        $p = ($aa * ($r - $b) + $bb) * ($r - $a) + $fa;
                        $pp = $bb + $aa * (2 * $r - $a - $b);
                        $r1 = ($pp != 0.0) ? $r - $p / $pp : $r;
                        if (!($a < $r1 && $r1 < $b)) {
                            break;
                        }
                        $r = $r1;
                    }
                    $c = $r;
                }
            } else {
                $c = $a - $fa * ($b - $a) / ($fb - $fa);
            }
            if (!($a < $c && $c < $b)) {
                $c = ($a + $b) / 2.0;
            }
            $fc = (float) $f($c);
            if ($fc == 0.0) {
                return $c;
            }
            $rebracket($c, $fc, $a, $fa, $b, $fb, $d, $fd);
            $haveD = true;
            // step 2: double-length secant from the endpoint with the smaller |f|
            $slope = ($fb - $fa) / ($b - $a);
            $u = (abs($fa) < abs($fb)) ? $a : $b;
            $fu = (abs($fa) < abs($fb)) ? $fa : $fb;
            $c = ($slope != 0.0) ? $u - 2.0 * $fu / $slope : ($a + $b) / 2.0;
            if (abs($c - $u) > 0.5 * ($b - $a) || !($a < $c && $c < $b)) {
                $c = ($a + $b) / 2.0;
            }
            $fc = (float) $f($c);
            if ($fc == 0.0) {
                return $c;
            }
            $rebracket($c, $fc, $a, $fa, $b, $fb, $d, $fd);
            // step 3: bisection safeguard if the interval did not shrink enough
            if (($b - $a) > 0.5 * $width) {
                $z = ($a + $b) / 2.0;
                $fz = (float) $f($z);
                if ($fz == 0.0) {
                    return $z;
                }
                $rebracket($z, $fz, $a, $fa, $b, $fb, $d, $fd);
            }
            if (($b - $a) < $xtol + $rtol * max(abs($a), abs($b))) {
                return (abs($fa) <= abs($fb)) ? $a : $b;
            }
        }

        throw new ConvergenceError("toms748: failed to converge after {$maxiter} iterations.");
    }

    /**
     * scipy.optimize.root_scalar: find a root of a scalar function by dispatching to a named solver. Bracketing
     * methods ('brentq', 'bisect', 'brenth', 'ridder') need a two-element bracket; derivative-free point methods
     * ('newton', 'secant') need a starting point x0. When method is null it is selected as scipy does: 'brentq'
     * if a bracket is given, else 'secant' if x1 is given, else 'newton'. Returns the result as
     * ['root' => float, 'converged' => bool, 'flag' => string].
     *
     * @param callable(float): float $f
     * @param list<float>|null $bracket
     * @return array{root: float, converged: bool, flag: string}
     */
    public static function rootScalar(callable $f, ?string $method = null, ?array $bracket = null, ?float $x0 = null, ?float $x1 = null): array
    {
        if ($method === null) {
            if ($bracket !== null) {
                $method = 'brentq';
            } elseif ($x0 !== null) {
                $method = ($x1 !== null) ? 'secant' : 'newton';
            } else {
                throw new TesseroException('root_scalar: provide a bracket or a starting point x0.');
            }
        }
        $m = strtolower($method);
        $converged = true;
        try {
            if (in_array($m, ['brentq', 'bisect', 'brenth', 'ridder'], true)) {
                if ($bracket === null || count($bracket) < 2) {
                    throw new TesseroException("root_scalar: method '{$method}' needs a two-element bracket.");
                }
                $a = (float) $bracket[0];
                $b = (float) $bracket[1];
                $root = match ($m) {
                    'brentq' => self::brentq($f, $a, $b),
                    'bisect' => self::bisect($f, $a, $b),
                    'brenth' => self::brenth($f, $a, $b),
                    'ridder' => self::ridder($f, $a, $b),
                };
            } elseif ($m === 'newton' || $m === 'secant') {
                if ($x0 === null) {
                    throw new TesseroException("root_scalar: method '{$method}' needs a starting point x0.");
                }
                $root = self::newton($f, $x0);
            } else {
                throw new TesseroException("root_scalar: unknown method '{$method}' (bisect, brentq, brenth, ridder, newton, secant).");
            }
        } catch (ConvergenceError) {
            return ['root' => NAN, 'converged' => false, 'flag' => 'convergence error'];
        }

        return ['root' => $root, 'converged' => $converged, 'flag' => 'converged'];
    }

    /**
     * Newton-Raphson (with fprime) or secant (without), as scipy.optimize.newton for scalars.
     *
     * @param callable(float): float $f
     * @param (callable(float): float)|null $fprime
     */
    public static function newton(callable $f, float $x0, ?callable $fprime = null, float $tol = 1.48e-8, int $maxiter = 50, float $rtol = 0.0): float
    {
        if ($fprime !== null) {
            $p0 = $x0;
            for ($i = 0; $i < $maxiter; $i++) {
                $fval = (float) $f($p0);
                if ($fval == 0.0) {
                    return $p0;
                }
                $d = (float) $fprime($p0);
                if ($d == 0.0) {
                    throw new ConvergenceError('newton: derivative was zero.');
                }
                $p = $p0 - $fval / $d;
                if (abs($p - $p0) <= $tol + $rtol * abs($p)) {
                    return $p;
                }
                $p0 = $p;
            }

            throw new ConvergenceError("newton: failed to converge after {$maxiter} iterations, value is {$p0}.");
        }
        // secant, scipy's starting points
        $eps = 1e-4;
        $p0 = $x0;
        $p1 = $x0 * (1 + $eps);
        $p1 += ($p1 >= 0 ? $eps : -$eps);
        $q0 = (float) $f($p0);
        $q1 = (float) $f($p1);
        if (abs($q1) < abs($q0)) {
            [$p0, $p1, $q0, $q1] = [$p1, $p0, $q1, $q0];
        }
        for ($i = 0; $i < $maxiter; $i++) {
            if ($q1 == $q0) {
                if ($p1 != $p0) {
                    throw new ConvergenceError('newton: secant tolerance reached with f(p0) == f(p1).');
                }

                return ($p1 + $p0) / 2.0;
            }
            if (abs($q1) > abs($q0)) {
                $p = (-$q0 / $q1 * $p1 + $p0) / (1 - $q0 / $q1);
            } else {
                $p = (-$q1 / $q0 * $p0 + $p1) / (1 - $q1 / $q0);
            }
            if (abs($p - $p1) <= $tol + $rtol * abs($p)) {
                return $p;
            }
            $p0 = $p1;
            $q0 = $q1;
            $p1 = $p;
            $q1 = (float) $f($p1);
        }

        throw new ConvergenceError("newton: failed to converge after {$maxiter} iterations, value is {$p1}.");
    }

    /**
     * Solve F(x) = 0 for a vector function with damped Newton steps
     * (Jacobian by forward differences unless given; linear solves by LAPACK
     * when available, partial-pivot Gaussian elimination otherwise).
     *
     * @param callable(list<float>): list<float> $fun
     * @param list<float> $x0
     * @param (callable(list<float>): list<list<float>>)|null $jac
     */
    /**
     * scipy.optimize.fsolve: find a root of a vector function, returning the solution vector (the default
     * full_output=0 behaviour). A thin wrapper over root(); for a well-posed system the root is unique, so it
     * matches scipy (whose fsolve uses MINPACK hybr).
     *
     * @param callable(list<float>): list<float> $func
     * @param list<float|int> $x0
     * @return list<float>
     */
    public static function fsolve(callable $func, array $x0): array
    {
        return self::root($func, array_map('floatval', array_values($x0)))->x;
    }

    /**
     * scipy.optimize.fixed_point: a scalar fixed point of func (func(x) = x) by Steffensen's method with Aitken's
     * del^2 acceleration, as scipy's default method='del2'. Matches scipy iterate for iterate.
     *
     * @param callable(float): float $func
     */
    public static function fixedPoint(callable $func, float $x0, float $xtol = 1e-8, int $maxiter = 500): float
    {
        $p0 = $x0;
        for ($i = 0; $i < $maxiter; $i++) {
            $p1 = (float) $func($p0);
            $p2 = (float) $func($p1);
            $d = $p2 - 2.0 * $p1 + $p0;
            $p = ($d == 0.0) ? $p2 : $p0 - ($p1 - $p0) ** 2 / $d;
            $relerr = ($p0 == 0.0) ? $p : ($p - $p0) / $p0;
            if (abs($relerr) < $xtol) {
                return $p;
            }
            $p0 = $p;
        }

        throw new ConvergenceError("fixed_point: failed to converge after {$maxiter} iterations.");
    }

    public static function root(callable $fun, array $x0, ?callable $jac = null, float $tol = 1.49012e-8, int $maxiter = 100): OptimizeResult
    {
        $x = array_map('floatval', array_values($x0));
        $n = count($x);
        $nfev = 0;
        $F = static function (array $v) use ($fun, &$nfev): array {
            $nfev++;

            return array_map('floatval', array_values($fun($v)));
        };
        $fx = $F($x);
        $norm = static fn (array $v): float => sqrt(Minimize::dot($v, $v));
        $it = 0;
        $status = 1;
        $msg = 'The iteration is not making good progress.';
        while ($it < $maxiter) {
            if ($norm($fx) <= $tol) {
                $status = 0;
                $msg = 'The solution converged.';
                break;
            }
            $J = $jac !== null ? $jac($x) : self::fdJacobian($F, $x, $fx);
            $step = Dense::solve($J, array_map(static fn (float $v): float => -$v, $fx));
            $t = 1.0;
            $f0 = $norm($fx);
            $improved = false;
            for ($ls = 0; $ls < 30; $ls++) {
                $xt = [];
                foreach ($x as $i => $v) {
                    $xt[] = $v + $t * $step[$i];
                }
                $ft = $F($xt);
                if ($norm($ft) < (1 - 1e-4 * $t) * $f0) {
                    $improved = true;
                    break;
                }
                $t *= 0.5;
            }
            $it++;
            if (! $improved) {
                break;
            }
            $dx = $t * $norm($step);
            $x = $xt;
            $fx = $ft;
            if ($dx <= $tol * (1 + $norm($x))) {
                $status = 0;
                $msg = 'The solution converged.';
                break;
            }
        }

        return new OptimizeResult($x, 0.5 * Minimize::dot($fx, $fx), $status === 0, $status, $msg, $it, $nfev, 0, $fx, null, 'newton');
    }

    /** @return list<list<float>> */
    public static function fdJacobian(callable $F, array $x, array $fx): array
    {
        $n = count($x);
        $m = count($fx);
        $J = array_fill(0, $m, array_fill(0, $n, 0.0));
        $rel = sqrt(self::EPS);
        foreach ($x as $j => $xj) {
            $h = $rel * max(1.0, abs($xj)) * ($xj >= 0 ? 1.0 : -1.0);
            $xh = $x;
            $xh[$j] = $xj + $h;
            $h = $xh[$j] - $xj;
            $fh = $F($xh);
            for ($i = 0; $i < $m; $i++) {
                $J[$i][$j] = ($fh[$i] - $fx[$i]) / $h;
            }
        }

        return $J;
    }
}
