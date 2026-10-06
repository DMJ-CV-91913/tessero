<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Exceptions\TesseroException;

/**
 * scipy.optimize.minimize for smooth problems of modest dimension, with the
 * objective written as a plain PHP callable over list<float>.
 *
 *   BFGS         scipy's algorithm step for step (inverse-Hessian update,
 *                strong-Wolfe line search from scalar_search_wolfe2)
 *   L-BFGS-B     limited-memory BFGS with box bounds (projected two-loop
 *                recursion; converges to the same KKT point as the Fortran code,
 *                intermediate iterates differ)
 *   Nelder-Mead  a line-by-line port of scipy's simplex (same iterates)
 *
 * Gradients default to scipy's 2-point finite differences.
 */
final class Minimize
{
    private const EPS = 2.220446049250313e-16;

    /** The Rosenbrock test function (scipy.optimize.rosen): sum 100 (x_{i+1}-x_i^2)^2 + (1-x_i)^2. */
    public static function rosen(array $x): float
    {
        $s = 0.0;
        for ($i = 0, $n = count($x); $i < $n - 1; $i++) {
            $s += 100.0 * ($x[$i + 1] - $x[$i] ** 2) ** 2 + (1.0 - $x[$i]) ** 2;
        }
        return $s;
    }

    /** Gradient of the Rosenbrock function (scipy.optimize.rosen_der). */
    public static function rosenDer(array $x): array
    {
        $n = count($x);
        $d = array_fill(0, $n, 0.0);
        if ($n < 2) { return $d; }
        $d[0] = -400.0 * $x[0] * ($x[1] - $x[0] ** 2) - 2.0 * (1.0 - $x[0]);
        for ($i = 1; $i < $n - 1; $i++) {
            $d[$i] = 200.0 * ($x[$i] - $x[$i - 1] ** 2) - 400.0 * ($x[$i + 1] - $x[$i] ** 2) * $x[$i] - 2.0 * (1.0 - $x[$i]);
        }
        $d[$n - 1] = 200.0 * ($x[$n - 1] - $x[$n - 2] ** 2);
        return $d;
    }

    /** Hessian matrix of the Rosenbrock function (scipy.optimize.rosen_hess). */
    public static function rosenHess(array $x): array
    {
        $n = count($x);
        $H = [];
        for ($i = 0; $i < $n; $i++) { $H[$i] = array_fill(0, $n, 0.0); }
        for ($i = 0; $i < $n - 1; $i++) { $H[$i][$i + 1] = -400.0 * $x[$i]; $H[$i + 1][$i] = -400.0 * $x[$i]; }
        $H[0][0] = 1200.0 * $x[0] ** 2 - 400.0 * $x[1] + 2.0;
        for ($i = 1; $i < $n - 1; $i++) { $H[$i][$i] = 202.0 + 1200.0 * $x[$i] ** 2 - 400.0 * $x[$i + 1]; }
        $H[$n - 1][$n - 1] += 200.0;
        return $H;
    }

    /** Hessian of the Rosenbrock function times a vector p (scipy.optimize.rosen_hess_prod). */
    public static function rosenHessProd(array $x, array $p): array
    {
        $n = count($x);
        $Hp = array_fill(0, $n, 0.0);
        $Hp[0] = (1200.0 * $x[0] ** 2 - 400.0 * $x[1] + 2.0) * $p[0] - 400.0 * $x[0] * $p[1];
        for ($i = 1; $i < $n - 1; $i++) {
            $Hp[$i] = -400.0 * $x[$i - 1] * $p[$i - 1] + (202.0 + 1200.0 * $x[$i] ** 2 - 400.0 * $x[$i + 1]) * $p[$i] - 400.0 * $x[$i] * $p[$i + 1];
        }
        $Hp[$n - 1] = -400.0 * $x[$n - 2] * $p[$n - 2] + 200.0 * $p[$n - 1];
        return $Hp;
    }

    /** scipy.optimize.brent: Brent's method on a downhill bracket, returning the minimiser (full_output=0). */
    public static function brent(callable $fun, ?array $brack = null): float
    {
        return self::minimizeScalar($fun, $brack, null, 'brent')->x[0];
    }

    /** scipy.optimize.golden: golden-section search, returning the minimiser (a port of _minimize_scalar_golden). */
    public static function golden(callable $fun, ?array $brack = null, float $tol = 1.4901161193847656e-08, int $maxiter = 5000): float
    {
        $gR = 0.61803399;
        $gC = 1.0 - $gR;
        if ($brack === null) {
            [$xa, $xb, $xc] = self::bracketMin($fun);
        } elseif (count($brack) === 2) {
            [$xa, $xb, $xc] = self::bracketMin($fun, (float) $brack[0], (float) $brack[1]);
        } else {
            $xa = (float) $brack[0]; $xb = (float) $brack[1]; $xc = (float) $brack[2];
        }
        $x0 = $xa; $x3 = $xc;
        if (abs($xc - $xb) > abs($xb - $xa)) { $x1 = $xb; $x2 = $xb + $gC * ($xc - $xb); }
        else { $x2 = $xb; $x1 = $xb - $gC * ($xb - $xa); }
        $f1 = (float) $fun($x1);
        $f2 = (float) $fun($x2);
        for ($i = 0; $i < $maxiter && abs($x3 - $x0) > $tol * (abs($x1) + abs($x2)); $i++) {
            if ($f2 < $f1) { $x0 = $x1; $x1 = $x2; $x2 = $gR * $x1 + $gC * $x3; $f1 = $f2; $f2 = (float) $fun($x2); }
            else { $x3 = $x2; $x2 = $x1; $x1 = $gR * $x2 + $gC * $x0; $f2 = $f1; $f1 = (float) $fun($x1); }
        }
        return $f1 < $f2 ? $x1 : $x2;
    }

    /** scipy.optimize.fmin_bfgs: minimise with BFGS and return the minimiser xopt. */
    public static function fminBfgs(callable $fun, array $x0, ?callable $fprime = null): array
    {
        return self::minimize($fun, $x0, $fprime, 'BFGS')->x;
    }

    private int $nfev = 0;

    private int $njev = 0;

    /** Nelder-Mead iteration counter: kept on the object, not in a local, see nelderMead(). */
    private int $nmIterations = 0;

    /** @var callable(list<float>): float */
    private $fun;

    /** @var (callable(list<float>): list<float>)|null */
    private $jac;

    private function __construct(callable $fun, ?callable $jac)
    {
        $this->fun = $fun;
        $this->jac = $jac;
    }

    /**
     * @param callable(list<float>): float $fun
     * @param list<float|int> $x0
     * @param (callable(list<float>): list<float>)|null $jac
     * @param list<array{0: float|null, 1: float|null}>|null $bounds
     * @param array<string, mixed> $options gtol, maxiter, maxcor, ftol, xatol, fatol, eps
     */
    public static function minimize(callable $fun, array $x0, ?callable $jac = null, string $method = 'BFGS', ?array $bounds = null, array $options = []): OptimizeResult
    {
        $x0 = array_map('floatval', array_values($x0));
        if ($x0 === []) {
            throw new TesseroException('minimize: x0 must not be empty.');
        }
        $self = new self($fun, $jac);
        $m = strtoupper($method);
        if ($bounds !== null && $m === 'BFGS') {
            $m = 'L-BFGS-B';
        }

        return match ($m) {
            'BFGS' => $self->bfgs($x0, $options),
            'L-BFGS-B', 'LBFGSB', 'L-BFGS' => $self->lbfgsb($x0, $bounds, $options),
            'NELDER-MEAD' => $self->nelderMead($x0, $options),
            default => throw new TesseroException("minimize: unknown method '{$method}' (BFGS, L-BFGS-B, Nelder-Mead)."),
        };
    }

    /**
     * scipy.optimize.fmin: minimise a scalar function with the Nelder-Mead simplex and return the minimiser
     * (the default full_output=False behaviour of scipy.optimize.fmin). Options: xtol, ftol, maxiter, maxfun.
     *
     * @param callable(list<float>): float $func
     * @param list<float|int> $x0
     * @param array<string, mixed> $options
     * @return list<float>
     */
    public static function fmin(callable $func, array $x0, array $options = []): array
    {
        $o = [];
        if (isset($options['xtol'])) {
            $o['xatol'] = $options['xtol'];
        }
        if (isset($options['ftol'])) {
            $o['fatol'] = $options['ftol'];
        }
        foreach (['maxiter', 'maxfun'] as $k) {
            if (isset($options[$k])) {
                $o[$k] = $options[$k];
            }
        }
        $o['xatol'] ??= 1e-4;                 // scipy.optimize.fmin defaults (xtol, ftol)
        $o['fatol'] ??= 1e-4;

        return self::minimize($func, $x0, null, 'Nelder-Mead', null, $o)->x;
    }

    /**
     * scipy.optimize.fminbound: bounded scalar minimisation by Brent's method with golden-section fallback - a
     * port of scipy's _minimize_scalar_bounded, returning the minimiser (the default full_output=0 behaviour).
     *
     * @param callable(float): float $func
     */
    public static function fminbound(callable $func, float $x1, float $x2, float $xtol = 1e-5, int $maxfun = 500): float
    {
        if ($x1 > $x2) {
            throw new TesseroException('fminbound: the lower bound exceeds the upper bound.');
        }
        $sqrtEps = sqrt(2.2e-16);
        $goldenMean = 0.5 * (3.0 - sqrt(5.0));
        $a = $x1;
        $b = $x2;
        $fulc = $a + $goldenMean * ($b - $a);
        $nfc = $xf = $fulc;
        $rat = $e = 0.0;
        $x = $xf;
        $fx = (float) $func($x);
        $num = 1;
        $ffulc = $fnfc = $fx;
        $xm = 0.5 * ($a + $b);
        $tol1 = $sqrtEps * abs($xf) + $xtol / 3.0;
        $tol2 = 2.0 * $tol1;
        while (abs($xf - $xm) > ($tol2 - 0.5 * ($b - $a))) {
            $golden = true;
            if (abs($e) > $tol1) {                           // try a parabolic fit
                $golden = false;
                $r = ($xf - $nfc) * ($fx - $ffulc);
                $q = ($xf - $fulc) * ($fx - $fnfc);
                $p = ($xf - $fulc) * $q - ($xf - $nfc) * $r;
                $q = 2.0 * ($q - $r);
                if ($q > 0.0) {
                    $p = -$p;
                }
                $q = abs($q);
                $r = $e;
                $e = $rat;
                if (abs($p) < abs(0.5 * $q * $r) && $p > $q * ($a - $xf) && $p < $q * ($b - $xf)) {
                    $rat = $p / $q;                          // parabolic step
                    $x = $xf + $rat;
                    if (($x - $a) < $tol2 || ($b - $x) < $tol2) {
                        $rat = (($xm - $xf) >= 0.0 ? 1.0 : -1.0) * $tol1;
                    }
                } else {
                    $golden = true;
                }
            }
            if ($golden) {                                   // golden-section step
                $e = ($xf >= $xm) ? ($a - $xf) : ($b - $xf);
                $rat = $goldenMean * $e;
            }
            $si = ($rat >= 0.0) ? 1.0 : -1.0;
            $x = $xf + $si * max(abs($rat), $tol1);
            $fu = (float) $func($x);
            $num++;
            if ($fu <= $fx) {
                if ($x >= $xf) {
                    $a = $xf;
                } else {
                    $b = $xf;
                }
                $fulc = $nfc;
                $ffulc = $fnfc;
                $nfc = $xf;
                $fnfc = $fx;
                $xf = $x;
                $fx = $fu;
            } else {
                if ($x < $xf) {
                    $a = $x;
                } else {
                    $b = $x;
                }
                if ($fu <= $fnfc || $nfc == $xf) {
                    $fulc = $nfc;
                    $ffulc = $fnfc;
                    $nfc = $x;
                    $fnfc = $fu;
                } elseif ($fu <= $ffulc || $fulc == $xf || $fulc == $nfc) {
                    $fulc = $x;
                    $ffulc = $fu;
                }
            }
            $xm = 0.5 * ($a + $b);
            $tol1 = $sqrtEps * abs($xf) + $xtol / 3.0;
            $tol2 = 2.0 * $tol1;
            if ($num >= $maxfun) {
                break;
            }
        }

        return $xf;
    }

    /**
     * scipy.optimize.minimize_scalar: minimise a function of one variable. Methods 'brent' (default; Brent's
     * method on a downhill bracket, a port of scipy's Brent class and bracket()) and 'bounded' (fminbound). The
     * result's x is a one-element list (Tessero's OptimizeResult carries a vector x).
     *
     * @param callable(float): float $func
     * @param list<float>|null $bracket   two or three initial points for 'brent'
     * @param array{0: float, 1: float}|null $bounds   interval for 'bounded'
     * @param array<string, mixed> $options   xtol/xatol, maxiter
     */
    public static function minimizeScalar(callable $func, ?array $bracket = null, ?array $bounds = null, string $method = '', array $options = []): OptimizeResult
    {
        $m = strtolower($method);
        if ($m === '') {
            $m = ($bounds !== null) ? 'bounded' : 'brent';
        }
        if ($m === 'brent') {
            $tol = (float) ($options['xtol'] ?? 1.48e-8);
            $maxiter = (int) ($options['maxiter'] ?? 500);
            [$xmin, $fval, $iter, $funcalls] = self::brentScalar($func, $bracket, $tol, $maxiter);

            return new OptimizeResult(x: [$xmin], fun: $fval, success: true, status: 0, message: 'Optimization terminated successfully.', nit: $iter, nfev: $funcalls);
        }
        if ($m === 'bounded') {
            if ($bounds === null || count($bounds) !== 2) {
                throw new TesseroException("minimize_scalar: the 'bounded' method requires two-element bounds.");
            }
            $xatol = (float) ($options['xatol'] ?? 1e-5);
            $maxfun = (int) ($options['maxiter'] ?? 500);
            $xf = self::fminbound($func, (float) $bounds[0], (float) $bounds[1], $xatol, $maxfun);
            $fx = (float) $func($xf);

            return new OptimizeResult(x: [$xf], fun: $fx, success: true, status: 0, message: 'Solution found.', nit: 0, nfev: 0);
        }

        throw new TesseroException("minimize_scalar: unknown method '{$method}' (brent, bounded).");
    }

    /**
     * scipy.optimize.bracket: bracket the minimum of a scalar function downhill from the initial points,
     * returning [xa, xb, xc, fa, fb, fc, funcalls] with fb <= fa and fb <= fc.
     *
     * @param callable(float): float $func
     * @return array{0: float, 1: float, 2: float, 3: float, 4: float, 5: float, 6: int}
     */
    public static function bracket(callable $func, float $xa = 0.0, float $xb = 1.0, float $growLimit = 110.0, int $maxiter = 1000): array
    {
        return self::bracketMin($func, $xa, $xb, $growLimit, $maxiter);
    }

    /**
     * Bracket a one-dimensional minimum (a port of scipy.optimize.bracket), returning
     * [xa, xb, xc, fa, fb, fc, funcalls] with fb <= fa and fb <= fc.
     *
     * @param callable(float): float $func
     * @return array{0: float, 1: float, 2: float, 3: float, 4: float, 5: float, 6: int}
     */
    private static function bracketMin(callable $func, float $xa = 0.0, float $xb = 1.0, float $growLimit = 110.0, int $maxiter = 1000): array
    {
        $gold = 1.618034;
        $verySmall = 1e-21;
        $fa = (float) $func($xa);
        $fb = (float) $func($xb);
        if ($fa < $fb) {
            [$xa, $xb] = [$xb, $xa];
            [$fa, $fb] = [$fb, $fa];
        }
        $xc = $xb + $gold * ($xb - $xa);
        $fc = (float) $func($xc);
        $funcalls = 3;
        $iter = 0;
        while ($fc < $fb) {
            $tmp1 = ($xb - $xa) * ($fb - $fc);
            $tmp2 = ($xb - $xc) * ($fb - $fa);
            $val = $tmp2 - $tmp1;
            $denom = (abs($val) < $verySmall) ? 2.0 * $verySmall : 2.0 * $val;
            $w = $xb - (($xb - $xc) * $tmp2 - ($xb - $xa) * $tmp1) / $denom;
            $wlim = $xb + $growLimit * ($xc - $xb);
            if ($iter > $maxiter) {
                throw new TesseroException('minimize_scalar: no valid bracket found before the iteration limit.');
            }
            $iter++;
            if (($w - $xc) * ($xb - $w) > 0.0) {
                $fw = (float) $func($w);
                $funcalls++;
                if ($fw < $fc) {
                    $xa = $xb;
                    $xb = $w;
                    $fa = $fb;
                    $fb = $fw;
                    break;
                }
                if ($fw > $fb) {
                    $xc = $w;
                    $fc = $fw;
                    break;
                }
                $w = $xc + $gold * ($xc - $xb);
                $fw = (float) $func($w);
                $funcalls++;
            } elseif (($w - $wlim) * ($wlim - $xc) >= 0.0) {
                $w = $wlim;
                $fw = (float) $func($w);
                $funcalls++;
            } elseif (($w - $wlim) * ($xc - $w) > 0.0) {
                $fw = (float) $func($w);
                $funcalls++;
                if ($fw < $fc) {
                    $xb = $xc;
                    $xc = $w;
                    $w = $xc + $gold * ($xc - $xb);
                    $fb = $fc;
                    $fc = $fw;
                    $fw = (float) $func($w);
                    $funcalls++;
                }
            } else {
                $w = $xc + $gold * ($xc - $xb);
                $fw = (float) $func($w);
                $funcalls++;
            }
            $xa = $xb;
            $xb = $xc;
            $xc = $w;
            $fa = $fb;
            $fb = $fc;
            $fc = $fw;
        }

        return [$xa, $xb, $xc, $fa, $fb, $fc, $funcalls];
    }

    /**
     * Brent's method for scalar minimisation (a port of scipy's Brent.optimize), returning
     * [xmin, fval, iterations, funcalls].
     *
     * @param callable(float): float $func
     * @param list<float>|null $brack
     * @return array{0: float, 1: float, 2: int, 3: int}
     */
    private static function brentScalar(callable $func, ?array $brack, float $tol, int $maxiter): array
    {
        $mintol = 1e-11;
        $cg = 0.3819660;
        if ($brack === null) {
            [$xa, $xb, $xc, , $fb, , $funcalls] = self::bracketMin($func);
        } elseif (count($brack) === 2) {
            [$xa, $xb, $xc, , $fb, , $funcalls] = self::bracketMin($func, (float) $brack[0], (float) $brack[1]);
        } else {
            $xa = (float) $brack[0];
            $xb = (float) $brack[1];
            $xc = (float) $brack[2];
            if ($xa > $xc) {
                [$xa, $xc] = [$xc, $xa];
            }
            $fb = (float) $func($xb);
            $funcalls = 3;
        }
        $x = $w = $v = $xb;
        $fw = $fv = $fx = $fb;                            // Brent seeds its state from f(xb), not a re-evaluation
        if ($xa < $xc) {
            $a = $xa;
            $b = $xc;
        } else {
            $a = $xc;
            $b = $xa;
        }
        $deltax = 0.0;
        $rat = 0.0;
        $iter = 0;
        while ($iter < $maxiter) {
            $tol1 = $tol * abs($x) + $mintol;
            $tol2 = 2.0 * $tol1;
            $xmid = 0.5 * ($a + $b);
            if (abs($x - $xmid) < ($tol2 - 0.5 * ($b - $a))) {
                break;
            }
            if (abs($deltax) <= $tol1) {
                $deltax = ($x >= $xmid) ? ($a - $x) : ($b - $x);
                $rat = $cg * $deltax;
            } else {
                $tmp1 = ($x - $w) * ($fx - $fv);
                $tmp2 = ($x - $v) * ($fx - $fw);
                $p = ($x - $v) * $tmp2 - ($x - $w) * $tmp1;
                $tmp2 = 2.0 * ($tmp2 - $tmp1);
                if ($tmp2 > 0.0) {
                    $p = -$p;
                }
                $tmp2 = abs($tmp2);
                $dxTemp = $deltax;
                $deltax = $rat;
                if ($p > $tmp2 * ($a - $x) && $p < $tmp2 * ($b - $x) && abs($p) < abs(0.5 * $tmp2 * $dxTemp)) {
                    $rat = $p / $tmp2;
                    $u = $x + $rat;
                    if (($u - $a) < $tol2 || ($b - $u) < $tol2) {
                        $rat = (($xmid - $x) >= 0) ? $tol1 : -$tol1;
                    }
                } else {
                    $deltax = ($x >= $xmid) ? ($a - $x) : ($b - $x);
                    $rat = $cg * $deltax;
                }
            }
            $u = (abs($rat) < $tol1) ? (($rat >= 0) ? ($x + $tol1) : ($x - $tol1)) : ($x + $rat);
            $fu = (float) $func($u);
            $funcalls++;
            if ($fu > $fx) {
                if ($u < $x) {
                    $a = $u;
                } else {
                    $b = $u;
                }
                if ($fu <= $fw || $w === $x) {
                    $v = $w;
                    $w = $u;
                    $fv = $fw;
                    $fw = $fu;
                } elseif ($fu <= $fv || $v === $x || $v === $w) {
                    $v = $u;
                    $fv = $fu;
                }
            } else {
                if ($u >= $x) {
                    $a = $x;
                } else {
                    $b = $x;
                }
                $v = $w;
                $w = $x;
                $x = $u;
                $fv = $fw;
                $fw = $fx;
                $fx = $fu;
            }
            $iter++;
        }

        return [$x, $fx, $iter, $funcalls];
    }

    /** scipy.optimize.approx_fprime-style forward difference gradient. */
    public static function approxGrad(callable $f, array $x, ?float $f0 = null, ?float $relStep = null): array
    {
        $f0 ??= (float) $f($x);
        $rel = $relStep ?? sqrt(self::EPS);
        $g = [];
        foreach ($x as $i => $xi) {
            $h = $rel * ($xi >= 0 ? 1.0 : -1.0) * max(1.0, abs($xi));
            $xh = $x;
            $xh[$i] = $xi + $h;
            $h = $xh[$i] - $xi;
            $g[$i] = ((float) $f($xh) - $f0) / $h;
        }

        return $g;
    }

    // ------------------------------------------------------------------ BFGS

    private function bfgs(array $x, array $o): OptimizeResult
    {
        $n = count($x);
        $gtol = (float) ($o['gtol'] ?? 1e-5);
        $maxiter = (int) ($o['maxiter'] ?? 200 * $n);
        $f = $this->f($x);
        $g = $this->g($x, $f);
        $H = self::identity($n);
        $oldOld = $f + self::norm2($g) / 2;
        $k = 0;
        $warn = 0;
        $msg = 'Optimization terminated successfully.';
        $gnorm = self::normInf($g);
        while ($gnorm > $gtol && $k < $maxiter) {
            $p = self::neg(self::matvec($H, $g));
            $ls = $this->wolfe($x, $p, $f, $oldOld, $g);
            if ($ls === null) {
                $warn = 2;
                $msg = 'Desired error not necessarily achieved due to precision loss.';
                break;
            }
            [$alpha, $fNew, $gNew] = $ls;
            $oldOld = $f;
            $f = $fNew;
            $xNew = self::axpy($alpha, $p, $x);
            $s = self::sub($xNew, $x);
            $x = $xNew;
            $gNew ??= $this->g($x, $f);
            $y = self::sub($gNew, $g);
            $g = $gNew;
            $k++;
            $gnorm = self::normInf($g);
            if ($gnorm <= $gtol) {
                break;
            }
            if (! is_finite($f)) {
                $warn = 2;
                $msg = 'Desired error not necessarily achieved due to precision loss.';
                break;
            }
            $ys = self::dot($y, $s);
            $rho = $ys == 0.0 ? 1000.0 : 1.0 / $ys;
            // H = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
            $Hy = self::matvec($H, $y);
            $yH = self::vecmat($y, $H);
            $yHy = self::dot($y, $Hy);
            $Hn = [];
            for ($i = 0; $i < $n; $i++) {
                $row = [];
                for ($j = 0; $j < $n; $j++) {
                    $row[] = $H[$i][$j] - $rho * ($s[$i] * $yH[$j] + $Hy[$i] * $s[$j]) + ($rho * $rho * $yHy + $rho) * $s[$i] * $s[$j];
                }
                $Hn[] = $row;
            }
            $H = $Hn;
        }
        if ($k >= $maxiter && $gnorm > $gtol) {
            $warn = 1;
            $msg = 'Maximum number of iterations has been exceeded.';
        }

        return new OptimizeResult($x, $f, $warn === 0, $warn, $msg, $k, $this->nfev, $this->njev, $g, $H, 'BFGS');
    }

    /**
     * Strong-Wolfe line search (scipy.optimize._linesearch.scalar_search_wolfe2).
     *
     * @return array{0: float, 1: float, 2: list<float>|null}|null
     */
    private function wolfe(array $x, array $p, float $phi0, float $oldPhi0, array $g0, float $c1 = 1e-4, float $c2 = 0.9): ?array
    {
        $derphi0 = self::dot($g0, $p);
        $gCache = [];
        $phi = function (float $a) use ($x, $p): float {
            return $this->f(self::axpy($a, $p, $x));
        };
        $derphi = function (float $a, float $fa) use ($x, $p, &$gCache): float {
            $gg = $this->g(self::axpy($a, $p, $x), $fa);
            $gCache = [$a, $gg];

            return self::dot($gg, $p);
        };
        $alpha1 = $derphi0 != 0.0 ? min(1.0, 1.01 * 2 * ($phi0 - $oldPhi0) / $derphi0) : 1.0;
        if ($alpha1 < 0) {
            $alpha1 = 1.0;
        }
        $alpha0 = 0.0;
        $phiA1 = $phi($alpha1);
        $phiA0 = $phi0;
        $derA0 = $derphi0;
        $result = null;
        for ($i = 0; $i < 10; $i++) {
            if ($alpha1 == 0.0) {
                break;
            }
            if ($phiA1 > $phi0 + $c1 * $alpha1 * $derphi0 || ($phiA1 >= $phiA0 && $i > 0)) {
                $result = $this->zoom($alpha0, $alpha1, $phiA0, $phiA1, $derA0, $phi, $derphi, $phi0, $derphi0, $c1, $c2);
                break;
            }
            $derA1 = $derphi($alpha1, $phiA1);
            if (abs($derA1) <= -$c2 * $derphi0) {
                $result = [$alpha1, $phiA1];
                break;
            }
            if ($derA1 >= 0) {
                $result = $this->zoom($alpha1, $alpha0, $phiA1, $phiA0, $derA1, $phi, $derphi, $phi0, $derphi0, $c1, $c2);
                break;
            }
            $alpha0 = $alpha1;
            $alpha1 = 2 * $alpha1;
            $phiA0 = $phiA1;
            $phiA1 = $phi($alpha1);
            $derA0 = $derA1;
        }
        if ($result === null) {
            return null;
        }
        [$a, $fa] = $result;
        $gg = ($gCache !== [] && $gCache[0] === $a) ? $gCache[1] : null;

        return [$a, $fa, $gg];
    }

    /** @return array{0: float, 1: float}|null */
    private function zoom(float $aLo, float $aHi, float $phiLo, float $phiHi, float $derLo, callable $phi, callable $derphi, float $phi0, float $derphi0, float $c1, float $c2): ?array
    {
        $phiRec = $phi0;
        $aRec = 0.0;
        for ($i = 0; $i <= 10; $i++) {
            $d = $aHi - $aLo;
            [$a, $b] = $d < 0 ? [$aHi, $aLo] : [$aLo, $aHi];
            $aj = null;
            if ($i > 0) {
                $aj = self::cubicmin($aLo, $phiLo, $derLo, $aHi, $phiHi, $aRec, $phiRec);
            }
            if ($i === 0 || $aj === null || $aj > $b - 0.2 * $d || $aj < $a + 0.2 * $d) {
                $aj = self::quadmin($aLo, $phiLo, $derLo, $aHi, $phiHi);
                if ($aj === null || $aj > $b - 0.1 * $d || $aj < $a + 0.1 * $d) {
                    $aj = $aLo + 0.5 * $d;
                }
            }
            $phiJ = $phi($aj);
            if ($phiJ > $phi0 + $c1 * $aj * $derphi0 || $phiJ >= $phiLo) {
                $phiRec = $phiHi;
                $aRec = $aHi;
                $aHi = $aj;
                $phiHi = $phiJ;
            } else {
                $derJ = $derphi($aj, $phiJ);
                if (abs($derJ) <= -$c2 * $derphi0) {
                    return [$aj, $phiJ];
                }
                if ($derJ * ($aHi - $aLo) >= 0) {
                    $phiRec = $phiHi;
                    $aRec = $aHi;
                    $aHi = $aLo;
                    $phiHi = $phiLo;
                } else {
                    $phiRec = $phiLo;
                    $aRec = $aLo;
                }
                $aLo = $aj;
                $phiLo = $phiJ;
                $derLo = $derJ;
            }
        }

        return null;
    }

    private static function cubicmin(float $a, float $fa, float $fpa, float $b, float $fb, float $c, float $fc): ?float
    {
        $C = $fpa;
        $db = $b - $a;
        $dc = $c - $a;
        $denom = ($db * $dc) ** 2 * ($db - $dc);
        if ($denom == 0.0) {
            return null;
        }
        $r1 = $fb - $fa - $C * $db;
        $r2 = $fc - $fa - $C * $dc;
        $A = ($dc ** 2 * $r1 - $db ** 2 * $r2) / $denom;
        $B = (-($dc ** 3) * $r1 + $db ** 3 * $r2) / $denom;
        $radical = $B * $B - 3 * $A * $C;
        if ($A == 0.0 || $radical < 0) {
            return null;
        }
        $x = $a + (-$B + sqrt($radical)) / (3 * $A);

        return is_finite($x) ? $x : null;
    }

    private static function quadmin(float $a, float $fa, float $fpa, float $b, float $fb): ?float
    {
        $db = $b - $a;
        if ($db == 0.0) {
            return null;
        }
        $B = ($fb - $fa - $fpa * $db) / ($db * $db);
        if ($B == 0.0) {
            return null;
        }
        $x = $a - $fpa / (2.0 * $B);

        return is_finite($x) ? $x : null;
    }

    // ------------------------------------------------------------------ L-BFGS-B (projected)

    private function lbfgsb(array $x, ?array $bounds, array $o): OptimizeResult
    {
        $n = count($x);
        $lo = array_fill(0, $n, -INF);
        $hi = array_fill(0, $n, INF);
        if ($bounds !== null) {
            if (count($bounds) !== $n) {
                throw new TesseroException('minimize: bounds needs one (min, max) pair per variable.');
            }
            foreach (array_values($bounds) as $i => $b) {
                $lo[$i] = $b[0] ?? -INF;
                $hi[$i] = $b[1] ?? INF;
            }
        }
        $m = (int) ($o['maxcor'] ?? 10);
        $pgtol = (float) ($o['gtol'] ?? 1e-5);
        $ftol = (float) ($o['ftol'] ?? 2.220446049250313e-09);
        $maxiter = (int) ($o['maxiter'] ?? 15000);
        $x = self::project($x, $lo, $hi);
        $f = $this->f($x);
        $g = $this->g($x, $f);
        $S = [];
        $Y = [];
        $k = 0;
        $status = 0;
        $msg = 'CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL';
        while (true) {
            $pg = self::projectedGradNorm($x, $g, $lo, $hi);
            if ($pg <= $pgtol) {
                break;
            }
            if ($k >= $maxiter) {
                $status = 1;
                $msg = 'STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT';
                break;
            }
            // two-loop recursion on the free variables
            $free = [];
            for ($i = 0; $i < $n; $i++) {
                $atLo = $x[$i] <= $lo[$i] && $g[$i] > 0;
                $atHi = $x[$i] >= $hi[$i] && $g[$i] < 0;
                $free[$i] = ! ($atLo || $atHi);
            }
            $q = [];
            for ($i = 0; $i < $n; $i++) {
                $q[$i] = $free[$i] ? $g[$i] : 0.0;
            }
            $alphas = [];
            for ($j = count($S) - 1; $j >= 0; $j--) {
                $rho = 1.0 / self::dot($Y[$j], $S[$j]);
                $a = $rho * self::dot($S[$j], $q);
                $alphas[$j] = $a;
                $q = self::axpy(-$a, $Y[$j], $q);
            }
            $gamma = $S !== [] ? self::dot($S[count($S) - 1], $Y[count($Y) - 1]) / self::dot($Y[count($Y) - 1], $Y[count($Y) - 1]) : 1.0 / max(1.0, self::norm2($g));
            $r = self::scale($gamma, $q);
            for ($j = 0; $j < count($S); $j++) {
                $rho = 1.0 / self::dot($Y[$j], $S[$j]);
                $beta = $rho * self::dot($Y[$j], $r);
                $r = self::axpy($alphas[$j] - $beta, $S[$j], $r);
            }
            $d = [];
            for ($i = 0; $i < $n; $i++) {
                $d[$i] = $free[$i] ? -$r[$i] : 0.0;
            }
            if (self::dot($d, $g) >= 0) {
                // not a descent direction: fall back to projected steepest descent
                $d = self::neg($g);
                $S = [];
                $Y = [];
            }
            // projected backtracking (Armijo) along the path P(x + t d)
            $t = 1.0;
            $accepted = false;
            for ($ls = 0; $ls < 40; $ls++) {
                $xt = self::project(self::axpy($t, $d, $x), $lo, $hi);
                $ft = $this->f($xt);
                $step = self::sub($xt, $x);
                if ($ft <= $f + 1e-4 * self::dot($g, $step)) {
                    $accepted = true;
                    break;
                }
                $t *= 0.5;
            }
            if (! $accepted) {
                $status = 2;
                $msg = 'ABNORMAL_TERMINATION_IN_LNSRCH';
                break;
            }
            $gt = $this->g($xt, $ft);
            $s = self::sub($xt, $x);
            $y = self::sub($gt, $g);
            $fPrev = $f;
            $x = $xt;
            $f = $ft;
            $g = $gt;
            $k++;
            if (self::dot($s, $y) > self::EPS * self::dot($y, $y)) {
                $S[] = $s;
                $Y[] = $y;
                if (count($S) > $m) {
                    array_shift($S);
                    array_shift($Y);
                }
            }
            if (($fPrev - $f) / max(abs($fPrev), abs($f), 1.0) <= $ftol) {
                $msg = 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH';
                break;
            }
        }

        return new OptimizeResult($x, $f, $status === 0, $status, $msg, $k, $this->nfev, $this->njev, $g, null, 'L-BFGS-B');
    }

    private static function project(array $x, array $lo, array $hi): array
    {
        foreach ($x as $i => $v) {
            $x[$i] = min(max($v, $lo[$i]), $hi[$i]);
        }

        return $x;
    }

    private static function projectedGradNorm(array $x, array $g, array $lo, array $hi): float
    {
        $m = 0.0;
        foreach ($x as $i => $v) {
            $pg = min(max($v - $g[$i], $lo[$i]), $hi[$i]) - $v;
            $m = max($m, abs($pg));
        }

        return $m;
    }

    // ------------------------------------------------------------------ Nelder-Mead (scipy port)

    private function nelderMead(array $x0, array $o): OptimizeResult
    {
        $n = count($x0);
        $maxiter = (int) ($o['maxiter'] ?? 200 * $n);
        $maxfev = (int) ($o['maxfev'] ?? 200 * $n);
        $xatol = (float) ($o['xatol'] ?? 1e-4);
        $fatol = (float) ($o['fatol'] ?? 1e-4);
        $adaptive = (bool) ($o['adaptive'] ?? false);
        // PHP 8.4's function JIT corrupted a local loop counter in this method (issue 4 in
        // docs/project/upstream-bugs.md; the trigger moves with unrelated edits). The counter
        // therefore lives in a property, and the coefficients use plain assignments.
        $rho = 1.0;
        $chi = $adaptive ? 1.0 + 2.0 / $n : 2.0;
        $psi = $adaptive ? 0.75 - 1.0 / (2.0 * $n) : 0.5;
        $sigma = $adaptive ? 1.0 - 1.0 / $n : 0.5;
        $sim = [$x0];
        for ($k = 0; $k < $n; $k++) {
            $y = $x0;
            $y[$k] = $y[$k] != 0.0 ? (1 + 0.05) * $y[$k] : 0.00025;
            $sim[] = $y;
        }
        $fsim = array_map(fn (array $v): float => $this->f($v), $sim);
        self::sortSimplex($sim, $fsim);
        $this->nmIterations = 1;
        while ($this->nfev < $maxfev && $this->nmIterations < $maxiter) {
            $xspread = 0.0;
            $fspread = 0.0;
            for ($k = 1; $k <= $n; $k++) {
                $fspread = max($fspread, abs($fsim[0] - $fsim[$k]));
                for ($j = 0; $j < $n; $j++) {
                    $xspread = max($xspread, abs($sim[$k][$j] - $sim[0][$j]));
                }
            }
            if ($xspread <= $xatol && $fspread <= $fatol) {
                break;
            }
            $this->nelderMeadStep($sim, $fsim, $n, $rho, $chi, $psi, $sigma);
            $this->nmIterations = $this->nmIterations + 1;
            self::sortSimplex($sim, $fsim);
        }
        $status = 0;
        $msg = 'Optimization terminated successfully.';
        if ($this->nfev >= $maxfev) {
            $status = 1;
            $msg = 'Maximum number of function evaluations has been exceeded.';
        } elseif ($this->nmIterations >= $maxiter) {
            $status = 2;
            $msg = 'Maximum number of iterations has been exceeded.';
        }

        return new OptimizeResult($sim[0], $fsim[0], $status === 0, $status, $msg, $this->nmIterations, $this->nfev, 0, null, null, 'Nelder-Mead');
    }


    /**
     * One reflection / expansion / contraction / shrink step.
     */
    private function nelderMeadStep(array &$sim, array &$fsim, int $n, float $rho, float $chi, float $psi, float $sigma): void
    {
        $xbar = array_fill(0, $n, 0.0);
        for ($k = 0; $k < $n; $k++) {
            for ($j = 0; $j < $n; $j++) {
                $xbar[$j] = $xbar[$j] + $sim[$k][$j];
            }
        }
        for ($j = 0; $j < $n; $j++) {
            $xbar[$j] = $xbar[$j] / $n;
        }
        $xr = self::lin(1 + $rho, $xbar, -$rho, $sim[$n]);
        $fxr = $this->f($xr);
        $doshrink = false;
        if ($fxr < $fsim[0]) {
            $xe = self::lin(1 + $rho * $chi, $xbar, -$rho * $chi, $sim[$n]);
            $fxe = $this->f($xe);
            if ($fxe < $fxr) {
                $sim[$n] = $xe;
                $fsim[$n] = $fxe;
            } else {
                $sim[$n] = $xr;
                $fsim[$n] = $fxr;
            }
        } elseif ($fxr < $fsim[$n - 1]) {
            $sim[$n] = $xr;
            $fsim[$n] = $fxr;
        } elseif ($fxr < $fsim[$n]) {
            $xc = self::lin(1 + $psi * $rho, $xbar, -$psi * $rho, $sim[$n]);
            $fxc = $this->f($xc);
            if ($fxc <= $fxr) {
                $sim[$n] = $xc;
                $fsim[$n] = $fxc;
            } else {
                $doshrink = true;
            }
        } else {
            $xcc = self::lin(1 - $psi, $xbar, $psi, $sim[$n]);
            $fxcc = $this->f($xcc);
            if ($fxcc < $fsim[$n]) {
                $sim[$n] = $xcc;
                $fsim[$n] = $fxcc;
            } else {
                $doshrink = true;
            }
        }
        if ($doshrink) {
            for ($j = 1; $j <= $n; $j++) {
                $sim[$j] = self::lin(1 - $sigma, $sim[0], $sigma, $sim[$j]);
                $fsim[$j] = $this->f($sim[$j]);
            }
        }
    }

    private static function sortSimplex(array &$sim, array &$fsim): void
    {
        $idx = array_keys($fsim);
        // stable sort by value, like numpy.argsort(kind='quicksort') on distinct values
        usort($idx, static fn (int $a, int $b): int => $fsim[$a] <=> $fsim[$b] ?: $a <=> $b);
        $sim = array_map(static fn (int $i): array => $sim[$i], $idx);
        $fsim = array_map(static fn (int $i): float => $fsim[$i], $idx);
    }

    private static function lin(float $a, array $x, float $b, array $y): array
    {
        $out = [];
        foreach ($x as $i => $v) {
            $out[] = $a * $v + $b * $y[$i];
        }

        return $out;
    }

    // ------------------------------------------------------------------ plumbing

    private function f(array $x): float
    {
        $this->nfev++;

        return (float) ($this->fun)($x);
    }

    private function g(array $x, ?float $fx = null): array
    {
        if ($this->jac !== null) {
            $this->njev++;

            return array_map('floatval', array_values(($this->jac)($x)));
        }
        $this->njev++;
        $f0 = $fx ?? $this->f($x);
        $g = [];
        $rel = sqrt(self::EPS);
        foreach ($x as $i => $xi) {
            $h = $rel * ($xi >= 0 ? 1.0 : -1.0) * max(1.0, abs($xi));
            $xh = $x;
            $xh[$i] = $xi + $h;
            $h = $xh[$i] - $xi;
            $g[] = ($this->f($xh) - $f0) / $h;
        }

        return $g;
    }

    private static function identity(int $n): array
    {
        $I = [];
        for ($i = 0; $i < $n; $i++) {
            $row = array_fill(0, $n, 0.0);
            $row[$i] = 1.0;
            $I[] = $row;
        }

        return $I;
    }

    private static function matvec(array $A, array $x): array
    {
        $out = [];
        foreach ($A as $row) {
            $out[] = self::dot($row, $x);
        }

        return $out;
    }

    private static function vecmat(array $x, array $A): array
    {
        $n = count($A[0]);
        $out = array_fill(0, $n, 0.0);
        foreach ($A as $i => $row) {
            $xi = $x[$i];
            for ($j = 0; $j < $n; $j++) {
                $out[$j] = $out[$j] + $xi * $row[$j];
            }
        }

        return $out;
    }

    public static function dot(array $a, array $b): float
    {
        $s = 0.0;
        foreach ($a as $i => $v) {
            $s += $v * $b[$i];
        }

        return $s;
    }

    private static function axpy(float $a, array $x, array $y): array
    {
        $out = [];
        foreach ($y as $i => $v) {
            $out[] = $v + $a * $x[$i];
        }

        return $out;
    }

    private static function sub(array $a, array $b): array
    {
        $out = [];
        foreach ($a as $i => $v) {
            $out[] = $v - $b[$i];
        }

        return $out;
    }

    private static function neg(array $a): array
    {
        return array_map(static fn (float $v): float => -$v, $a);
    }

    private static function scale(float $s, array $a): array
    {
        return array_map(static fn (float $v): float => $s * $v, $a);
    }

    private static function norm2(array $a): float
    {
        return sqrt(self::dot($a, $a));
    }

    private static function normInf(array $a): float
    {
        return max(array_map('abs', $a));
    }
}
