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

    /** scipy.optimize.fmin_cg: nonlinear conjugate-gradient (Polak-Ribiere+) minimiser. */
    public static function fminCg(callable $fun, array $x0, ?callable $fprime = null): array
    {
        return self::minimize($fun, $x0, $fprime, 'CG')->x;
    }

    /** scipy.optimize.fmin_powell: derivative-free direction-set minimiser. */
    public static function fminPowell(callable $func, array $x0, array $options = []): array
    {
        return self::minimize($func, $x0, null, 'Powell', null, $options)->x;
    }

    /** scipy.optimize.fmin_ncg: truncated-Newton (Newton-CG) minimiser; fprime required, hessp optional. */
    public static function fminNcg(callable $f, array $x0, callable $fprime, ?callable $fhessp = null): array
    {
        return self::minimize($f, $x0, $fprime, 'Newton-CG', null, $fhessp !== null ? ['hessp' => $fhessp] : [])->x;
    }

    /**
     * scipy.optimize.fmin_tnc: bound-constrained minimisation by a truncated-Newton algorithm. Returns the
     * minimiser x (scipy returns (x, nfeval, rc); here we return x only, matching the other fmin_* wrappers).
     *
     * @param callable(list<float>): float $func
     * @param list<float> $x0
     * @param (callable(list<float>): list<float>)|null $fprime analytic gradient; finite-differenced if null
     * @param list<array{0: float|null, 1: float|null}>|null $bounds
     */
    public static function fminTnc(callable $func, array $x0, ?callable $fprime = null, ?array $bounds = null): array
    {
        return self::minimize($func, $x0, $fprime, 'TNC', $bounds)->x;
    }

    /**
     * scipy.optimize.fmin_l_bfgs_b: bound-constrained L-BFGS-B minimiser. scipy returns
     * (x, f, d); here we return the minimiser x only, matching the other fmin_* wrappers.
     *
     * @param callable(list<float>): float $func
     * @param list<float> $x0
     * @param (callable(list<float>): list<float>)|null $fprime analytic gradient; finite-differenced if null
     * @param list<array{0: float|null, 1: float|null}>|null $bounds
     */
    public static function fminLbfgsb(callable $func, array $x0, ?callable $fprime = null, ?array $bounds = null): array
    {
        return self::minimize($func, $x0, $fprime, 'L-BFGS-B', $bounds)->x;
    }

    /**
     * scipy.optimize.fmin_slsqp: sequential least-squares programming with equality / inequality constraints
     * and bounds. eqcons / ieqcons are lists of callables c(x) returning a scalar (eq: c==0, ineq: c>=0),
     * exactly as scipy's fmin_slsqp. Returns the minimiser x.
     *
     * @param callable(list<float>): float $func
     * @param list<float> $x0
     * @param list<callable(list<float>): float> $eqcons
     * @param list<callable(list<float>): float> $ieqcons
     * @param list<array{0: float|null, 1: float|null}>|null $bounds
     * @param (callable(list<float>): list<float>)|null $fprime
     */
    public static function fminSlsqp(callable $func, array $x0, array $eqcons = [], array $ieqcons = [], ?array $bounds = null, ?callable $fprime = null): array
    {
        $cons = [];
        foreach ($eqcons as $c) {
            $cons[] = ['type' => 'eq', 'fun' => $c];
        }
        foreach ($ieqcons as $c) {
            $cons[] = ['type' => 'ineq', 'fun' => $c];
        }

        return self::minimize($func, $x0, $fprime, 'SLSQP', $bounds, ['constraints' => $cons])->x;
    }

    /**
     * scipy.optimize.fmin_cobyla: constrained minimisation by linear approximation (Powell's COBYLA). cons is
     * a list of callables c(x) >= 0 (inequality constraints only, as in scipy). Returns the minimiser x.
     *
     * @param callable(list<float>): float $func
     * @param list<float> $x0
     * @param list<callable(list<float>): float> $cons
     */
    public static function fminCobyla(callable $func, array $x0, array $cons, float $rhobeg = 1.0, float $rhoend = 1e-4, int $maxfun = 1000): array
    {
        $c = [];
        foreach ($cons as $cc) {
            $c[] = ['type' => 'ineq', 'fun' => $cc];
        }

        return self::minimize($func, $x0, null, 'COBYLA', null, ['constraints' => $c, 'rhobeg' => $rhobeg, 'rhoend' => $rhoend, 'maxfun' => $maxfun])->x;
    }

    /**
     * scipy.optimize.check_grad: the 2-norm of the difference between the analytic gradient and a
     * forward-difference approximation of it at x0 (absolute step epsilon, as in scipy).
     *
     * @param callable(list<float>): float $func
     * @param callable(list<float>): list<float> $grad
     * @param list<float> $x0
     */
    public static function checkGrad(callable $func, callable $grad, array $x0, float $epsilon = 1.4901161193847656e-08): float
    {
        $x0 = array_map('floatval', array_values($x0));
        $g = $grad($x0);
        $f0 = (float) $func($x0);
        $n = count($x0);
        $sum = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $xp = $x0;
            $xp[$i] += $epsilon;
            $fd = ((float) $func($xp) - $f0) / $epsilon;
            $d = (float) $g[$i] - $fd;
            $sum += $d * $d;
        }

        return sqrt($sum);
    }

    /**
     * scipy.optimize.brute (finish=None): minimise by evaluating func on a regular grid over the given
     * ranges (Ns points per dimension, inclusive) and returning the grid point with the smallest value.
     *
     * @param callable(list<float>): float $func
     * @param list<array{0: float, 1: float}> $ranges
     * @return float|list<float>
     */
    public static function brute(callable $func, array $ranges, int $Ns = 20): float|array
    {
        $dims = count($ranges);
        $axes = [];
        foreach ($ranges as [$lo, $hi]) {
            $ax = [];
            for ($j = 0; $j < $Ns; $j++) {
                $ax[$j] = (float) $lo + ((float) $hi - (float) $lo) * $j / ($Ns - 1);
            }
            $axes[] = $ax;
        }
        $best = INF;
        $bestpt = array_fill(0, $dims, 0.0);
        $total = (int) ($Ns ** $dims);
        for ($idx = 0; $idx < $total; $idx++) {
            $pt = [];
            $t = $idx;
            for ($d = 0; $d < $dims; $d++) {
                $pt[$d] = $axes[$d][$t % $Ns];
                $t = intdiv($t, $Ns);
            }
            $v = (float) $func($pt);
            if ($v < $best) {
                $best = $v;
                $bestpt = $pt;
            }
        }

        return $dims === 1 ? $bestpt[0] : $bestpt;
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
            'CG' => $self->cg($x0, $options),
            'POWELL' => $self->powell($x0, $options),
            'NEWTON-CG', 'NEWTONCG' => $self->newtonCg($x0, $options),
            'TNC' => $self->tnc($x0, $bounds, $options),
            'SLSQP' => $self->slsqp($x0, $bounds, $options),
            'COBYLA' => $self->cobyla($x0, $options),
            'DOGLEG' => $self->trustRegion($x0, $options, 'dogleg'),
            'TRUST-NCG', 'TRUSTNCG' => $self->trustRegion($x0, $options, 'trust-ncg'),
            'TRUST-EXACT', 'TRUSTEXACT' => $self->trustRegion($x0, $options, 'trust-exact'),
            'TRUST-KRYLOV', 'TRUSTKRYLOV' => $self->trustRegion($x0, $options, 'trust-krylov'),
            default => throw new TesseroException("minimize: unknown method '{$method}' (BFGS, L-BFGS-B, Nelder-Mead, CG, Powell, Newton-CG, TNC, SLSQP, COBYLA)."),
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

    // ------------------------------------------------------------------ CG (Polak-Ribiere+)

    private function cg(array $x, array $o): OptimizeResult
    {
        $n = count($x);
        $gtol = (float) ($o['gtol'] ?? 1e-5);
        $maxiter = (int) ($o['maxiter'] ?? 200 * $n);
        $f = $this->f($x);
        $gfk = $this->g($x, $f);
        $oldOld = $f + self::norm2($gfk) / 2;
        $pk = self::neg($gfk);
        $k = 0;
        $warn = 0;
        $msg = 'Optimization terminated successfully.';
        $gnorm = self::normInf($gfk);
        $sigma3 = 0.01;
        while ($gnorm > $gtol && $k < $maxiter) {
            $deltak = self::dot($gfk, $gfk);
            $pkLocal = $pk;
            $xLocal = $x;
            $gfkLocal = $gfk;
            // scipy descent_condition: accept only if the NEXT PR+ direction is a sufficient descent direction
            $descent = function (float $alpha, float $phival) use ($xLocal, $pkLocal, $gfkLocal, $deltak, $sigma3, $gtol): bool {
                $gkp1 = $this->g(self::axpy($alpha, $pkLocal, $xLocal), $phival);
                if (self::normInf($gkp1) <= $gtol) {
                    return true;
                }
                $yk = self::sub($gkp1, $gfkLocal);
                $beta = max(0.0, self::dot($yk, $gkp1) / $deltak);
                $pkp1 = self::axpy($beta, $pkLocal, self::neg($gkp1));
                return self::dot($gkp1, $pkp1) <= -$sigma3 * self::dot($gkp1, $gkp1);
            };
            $ls = $this->wolfe($x, $pk, $f, $oldOld, $gfk, 1e-4, 0.4, $descent);
            if ($ls === null) {
                $warn = 2;
                $msg = 'Desired error not necessarily achieved due to precision loss.';
                break;
            }
            [$alpha, $fNew, $gNew] = $ls;
            $oldOld = $f;
            $f = $fNew;
            $x = self::axpy($alpha, $pk, $x);
            $gNew ??= $this->g($x, $f);
            $yk = self::sub($gNew, $gfk);
            $beta = max(0.0, self::dot($yk, $gNew) / $deltak);
            $pk = self::axpy($beta, $pk, self::neg($gNew));
            $gfk = $gNew;
            $k++;
            $gnorm = self::normInf($gfk);
            if (! is_finite($f)) {
                $warn = 2;
                $msg = 'Desired error not necessarily achieved due to precision loss.';
                break;
            }
        }
        if ($k >= $maxiter && $gnorm > $gtol) {
            $warn = 1;
            $msg = 'Maximum number of iterations has been exceeded.';
        }

        return new OptimizeResult($x, $f, $warn === 0, $warn, $msg, $k, $this->nfev, $this->njev, $gfk, null, 'CG');
    }

    // ------------------------------------------------------------------ Powell (direction set)

    private function powell(array $x, array $o): OptimizeResult
    {
        $n = count($x);
        $xtol = (float) ($o['xtol'] ?? 1e-4);
        $ftol = (float) ($o['ftol'] ?? 1e-4);
        $maxiter = (int) ($o['maxiter'] ?? 1000 * $n);
        $maxfun = (int) ($o['maxfun'] ?? 1000 * $n);
        $direc = self::identity($n);
        $fval = $this->f($x);
        $x1 = $x;
        $it = 0;
        $warn = 0;
        $msg = 'Optimization terminated successfully.';
        while (true) {
            $fx = $fval;
            $bigind = 0;
            $delta = 0.0;
            for ($i = 0; $i < $n; $i++) {
                $direc1 = $direc[$i];
                $fx2 = $fval;
                [$fval, $x, $direc1] = $this->linesearchPowell($x, $direc1, $xtol * 100);
                if (($fx2 - $fval) > $delta) {
                    $delta = $fx2 - $fval;
                    $bigind = $i;
                }
            }
            $it++;
            if (2.0 * ($fx - $fval) <= $ftol * (abs($fx) + abs($fval)) + 1e-20) {
                break;
            }
            if ($this->nfev >= $maxfun) {
                $warn = 1;
                $msg = 'Maximum number of function evaluations has been exceeded.';
                break;
            }
            if ($it >= $maxiter) {
                $warn = 1;
                $msg = 'Maximum number of iterations has been exceeded.';
                break;
            }
            if (is_nan($fx) && is_nan($fval)) {
                break;
            }
            $direc1 = self::sub($x, $x1);
            $x1 = $x;
            $x2 = self::axpy(1.0, $direc1, $x);
            $fx2 = $this->f($x2);
            if ($fx > $fx2) {
                $t = 2.0 * ($fx + $fx2 - 2.0 * $fval);
                $temp = $fx - $fval - $delta;
                $t *= $temp * $temp;
                $temp = $fx - $fx2;
                $t -= $delta * $temp * $temp;
                if ($t < 0.0) {
                    [$fval, $x, $direc1] = $this->linesearchPowell($x, $direc1, $xtol * 100);
                    if (self::anyNonzero($direc1)) {
                        $direc[$bigind] = $direc[$n - 1];
                        $direc[$n - 1] = $direc1;
                    }
                }
            }
        }

        return new OptimizeResult($x, $fval, $warn === 0, $warn, $msg, $it, $this->nfev, 0, null, null, 'Powell');
    }

    /** @return array{0: float, 1: list<float>, 2: list<float>} [fret, p+xi, xi] */
    private function linesearchPowell(array $p, array $xi, float $tol): array
    {
        if (! self::anyNonzero($xi)) {
            return [$this->f($p), $p, $xi];
        }
        $myfunc = fn (float $a): float => $this->f(self::axpy($a, $xi, $p));
        [$alphaMin, $fret] = self::brentScalar($myfunc, null, $tol, 500);
        $xi = self::scale($alphaMin, $xi);

        return [$fret, self::axpy(1.0, $xi, $p), $xi];
    }

    // ------------------------------------------------------------------ Newton-CG (truncated Newton)

    private function newtonCg(array $x, array $o): OptimizeResult
    {
        $n = count($x);
        $hessp = $o['hessp'] ?? null;
        $xtol = (float) ($o['xtol'] ?? 1e-5);
        $maxiter = (int) ($o['maxiter'] ?? 200 * $n);
        $cgMax = 20 * $n;
        $xtolEff = $n * $xtol;
        $eps = 1.4901161193847656e-08;
        $f64eps = 2.220446049250313e-16;
        $updateL1 = PHP_FLOAT_MAX;
        $f = $this->f($x);
        $oldOld = null;
        $k = 0;
        $warn = 0;
        $msg = 'Optimization terminated successfully.';
        while ($updateL1 > $xtolEff) {
            if ($k >= $maxiter) {
                $warn = 1;
                $msg = 'Maximum number of iterations has been exceeded.';
                break;
            }
            $g0 = $this->g($x, $f);
            $b = self::neg($g0);
            $maggrad = self::l1($b);
            $eta = min(0.5, sqrt($maggrad));
            $termcond = $eta * $maggrad;
            $xsupi = array_fill(0, $n, 0.0);
            $ri = self::neg($b);
            $psupi = self::neg($ri);
            $i = 0;
            $dri0 = self::dot($ri, $ri);
            $cgConverged = false;
            for ($j = 0; $j < $cgMax; $j++) {
                if (self::l1($ri) <= $termcond) {
                    $cgConverged = true;
                    break;
                }
                if ($hessp !== null) {
                    $Ap = array_map('floatval', array_values(($hessp)($x, $psupi)));
                } else {
                    $x1 = self::axpy($eps, $psupi, $x);
                    $g1 = $this->g($x1, $this->f($x1));
                    $Ap = [];
                    for ($t = 0; $t < $n; $t++) {
                        $Ap[$t] = ($g1[$t] - $g0[$t]) / $eps;
                    }
                }
                $curv = self::dot($psupi, $Ap);
                if ($curv >= 0 && $curv <= 3 * $f64eps) {
                    $cgConverged = true;
                    break;
                }
                if ($curv < 0) {
                    if ($i > 0) {
                        $cgConverged = true;
                        break;
                    }
                    $xsupi = self::scale($dri0 / (-$curv), $b);
                    $cgConverged = true;
                    break;
                }
                $alphai = $dri0 / $curv;
                $xsupi = self::axpy($alphai, $psupi, $xsupi);
                $ri = self::axpy($alphai, $Ap, $ri);
                $dri1 = self::dot($ri, $ri);
                $betai = $dri1 / $dri0;
                $psupi = self::axpy($betai, $psupi, self::neg($ri));
                $i++;
                $dri0 = $dri1;
            }
            if (! $cgConverged) {
                $warn = 2;
                $msg = 'Warning: CG iterations did not converge.';
                break;
            }
            $pk = $xsupi;
            $gfk = self::neg($b);
            $ls = $this->wolfe($x, $pk, $f, $oldOld ?? ($f + self::norm2($gfk) / 2), $gfk, 1e-4, 0.9);
            if ($ls === null) {
                $warn = 2;
                $msg = 'Line search failed.';
                break;
            }
            [$alphak, $fNew] = $ls;
            $oldOld = $f;
            $f = $fNew;
            $update = self::scale($alphak, $pk);
            $x = self::axpy(1.0, $update, $x);
            $k++;
            $updateL1 = self::l1($update);
        }

        return new OptimizeResult($x, $f, $warn === 0, $warn, $msg, $k, $this->nfev, $this->njev, null, null, 'Newton-CG');
    }

    private static function l1(array $a): float
    {
        $s = 0.0;
        foreach ($a as $v) {
            $s += abs($v);
        }

        return $s;
    }

    private static function anyNonzero(array $a): bool
    {
        foreach ($a as $v) {
            if ($v != 0.0) {
                return true;
            }
        }

        return false;
    }

    /**
     * Strong-Wolfe line search (scipy.optimize._linesearch.scalar_search_wolfe2).
     *
     * @return array{0: float, 1: float, 2: list<float>|null}|null
     */
    private function wolfe(array $x, array $p, float $phi0, float $oldPhi0, array $g0, float $c1 = 1e-4, float $c2 = 0.9, ?callable $extra = null): ?array
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
                $result = $this->zoom($alpha0, $alpha1, $phiA0, $phiA1, $derA0, $phi, $derphi, $phi0, $derphi0, $c1, $c2, $extra);
                break;
            }
            $derA1 = $derphi($alpha1, $phiA1);
            if (abs($derA1) <= -$c2 * $derphi0 && ($extra === null || $extra($alpha1, $phiA1))) {
                $result = [$alpha1, $phiA1];
                break;
            }
            if ($derA1 >= 0) {
                $result = $this->zoom($alpha1, $alpha0, $phiA1, $phiA0, $derA1, $phi, $derphi, $phi0, $derphi0, $c1, $c2, $extra);
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
    private function zoom(float $aLo, float $aHi, float $phiLo, float $phiHi, float $derLo, callable $phi, callable $derphi, float $phi0, float $derphi0, float $c1, float $c2, ?callable $extra = null): ?array
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
                if (abs($derJ) <= -$c2 * $derphi0 && ($extra === null || $extra($aj, $phiJ))) {
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

    // ------------------------------------------------------------------ TNC (truncated Newton, box-bounded)

    /**
     * scipy.optimize.minimize(method='TNC'): a truncated-Newton algorithm with box bounds. Each outer step
     * solves H d = -g on the free variables by conjugate gradients (Hessian-vector products from a
     * finite-difference of the gradient), then a projected Armijo line search clips to the bounds. Converges
     * to scipy's TNC optimum; the intermediate iterate sequence differs (scipy wraps Nash's Fortran tnc).
     *
     * @param list<float> $x
     * @param list<array{0: float|null, 1: float|null}>|null $bounds
     * @param array<string, mixed> $o
     */
    private function tnc(array $x, ?array $bounds, array $o): OptimizeResult
    {
        $n = count($x);
        [$lo, $hi] = self::unpackBounds($bounds, $n);
        $gtol = (float) ($o['gtol'] ?? 1e-6);
        $maxiter = (int) ($o['maxiter'] ?? 200);
        $x = self::project($x, $lo, $hi);
        $f = $this->f($x);
        $status = 1;
        $msg = 'Local minimum reached (|pg| ~= 0)';
        $k = 0;
        $g = $this->g($x, $f);
        for ($outer = 0; $outer < 4 * $maxiter; $outer++) {
            $g = $this->g($x);
            // free set: a variable is fixed only if it sits on a bound with the gradient pushing outward.
            $free = [];
            for ($i = 0; $i < $n; $i++) {
                $atLo = $x[$i] <= $lo[$i] + 1e-12 && $g[$i] > 0;
                $atHi = $x[$i] >= $hi[$i] - 1e-12 && $g[$i] < 0;
                $free[$i] = ! ($atLo || $atHi);
            }
            $pg = [];
            for ($i = 0; $i < $n; $i++) {
                $pg[$i] = $free[$i] ? $g[$i] : 0.0;
            }
            if (self::normInf($pg) <= $gtol) {
                $status = 0;
                break;
            }
            // CG to approximately solve H d = -pg on the free subspace
            $d = array_fill(0, $n, 0.0);
            $r = self::neg($pg);
            $p = $r;
            $rs = self::dot($r, $r);
            $pgnorm = sqrt(self::dot($pg, $pg));
            $maxcg = 2 * $n + 2;
            for ($j = 0; $j < $maxcg; $j++) {
                $pf = [];
                for ($i = 0; $i < $n; $i++) {
                    $pf[$i] = $free[$i] ? $p[$i] : 0.0;
                }
                $Hp = $this->hessVec($x, $pf, $g);
                for ($i = 0; $i < $n; $i++) {
                    $Hp[$i] = $free[$i] ? $Hp[$i] : 0.0;
                }
                $pHp = self::dot($p, $Hp);
                if ($pHp <= 1e-14) {
                    if (! self::anyNonzero($d)) {
                        $d = $r;
                    }
                    break;
                }
                $alpha = $rs / $pHp;
                $d = self::axpy($alpha, $p, $d);
                $r = self::axpy(-$alpha, $Hp, $r);
                $rs2 = self::dot($r, $r);
                if (sqrt($rs2) < 0.1 * max($pgnorm, 1e-15)) {
                    break;
                }
                $p = self::axpy($rs2 / $rs, $p, $r);
                $rs = $rs2;
            }
            for ($i = 0; $i < $n; $i++) {
                $d[$i] = $free[$i] ? $d[$i] : 0.0;
            }
            if (self::dot($d, $g) >= 0) {
                $d = self::neg($pg);
            }
            // projected backtracking (Armijo) along P(x + alpha d)
            $alpha = 1.0;
            $accepted = false;
            $xn = $x;
            for ($ls = 0; $ls < 40; $ls++) {
                $xn = self::project(self::axpy($alpha, $d, $x), $lo, $hi);
                $ft = $this->f($xn);
                $step = self::sub($xn, $x);
                if ($ft <= $f + 1e-4 * self::dot($g, $step)) {
                    $accepted = true;
                    break;
                }
                $alpha *= 0.5;
            }
            if (! $accepted) {
                break;
            }
            if (self::norm2(self::sub($xn, $x)) < 1e-12) {
                $x = $xn;
                $f = $this->f($x);
                break;
            }
            $x = $xn;
            $f = $this->f($x);
            $k++;
        }

        return new OptimizeResult($x, $f, $status === 0, $status, $msg, $k, $this->nfev, $this->njev, $g, null, 'TNC');
    }

    /** Finite-difference Hessian-vector product (df-of-gradient), as TNC/Newton-CG use. */
    private function hessVec(array $x, array $v, array $g): array
    {
        $eps = 1e-7;
        $xv = self::axpy($eps, $v, $x);
        $gv = $this->g($xv);

        return array_map(static fn (float $a, float $b): float => ($a - $b) / $eps, $gv, $g);
    }

    // ------------------------------------------------------------------ SLSQP (Kraft sequential QP)

    /**
     * scipy.optimize.minimize(method='SLSQP'): sequential least-squares programming. At each iterate we solve
     * the QP  min 1/2 d'Bd + g'd  s.t. linearised equality / inequality constraints and the box bounds, with B
     * a BFGS approximation of the Lagrangian Hessian, then take an L1-merit line search. Converges to the same
     * KKT point as scipy's slsqp (Kraft's Fortran); intermediate iterates differ.
     *
     * @param list<float> $x
     * @param list<array{0: float|null, 1: float|null}>|null $bounds
     * @param array<string, mixed> $o constraints: list<array{type:'eq'|'ineq', fun:callable}>
     */
    private function slsqp(array $x, ?array $bounds, array $o): OptimizeResult
    {
        $n = count($x);
        [$lo, $hi] = self::unpackBounds($bounds, $n);
        $cons = $o['constraints'] ?? [];
        $eq = [];
        $ineq = [];
        foreach ($cons as $c) {
            if (($c['type'] ?? 'ineq') === 'eq') {
                $eq[] = $c['fun'];
            } else {
                $ineq[] = $c['fun'];
            }
        }
        $tol = (float) ($o['ftol'] ?? 1e-6);
        $maxiter = (int) ($o['maxiter'] ?? 200);
        $B = self::identity($n);
        $status = 1;
        $msg = 'Iteration limit reached';
        $it = 0;
        $f = $this->f($x);
        for ($it = 0; $it < $maxiter; $it++) {
            $gk = $this->g($x);
            $Ceq = [];
            $beq = [];
            foreach ($eq as $c) {
                $Ceq[] = self::constraintGrad($c, $x);
                $beq[] = (float) $c($x);
            }
            $Cin = [];
            $bin = [];
            foreach ($ineq as $c) {
                $Cin[] = self::constraintGrad($c, $x);
                $bin[] = (float) $c($x);
            }
            $d = self::solveQp($B, $gk, $Ceq, $beq, $Cin, $bin, $lo, $hi, $x);
            $nd = self::norm2($d);
            if ($nd < $tol) {
                $status = 0;
                $msg = 'Optimization terminated successfully';
                break;
            }
            // L1 merit line search
            $mu = 10.0;
            $merit = function (array $z) use ($eq, $ineq, $mu): float {
                $viol = 0.0;
                foreach ($eq as $c) {
                    $viol += abs((float) $c($z));
                }
                foreach ($ineq as $c) {
                    $viol += max(0.0, -(float) $c($z));
                }

                return $this->f($z) + $mu * $viol;
            };
            $m0 = $merit($x);
            $alpha = 1.0;
            $xn = $x;
            for ($ls = 0; $ls < 30; $ls++) {
                $xn = self::project(self::axpy($alpha, $d, $x), $lo, $hi);
                if ($merit($xn) <= $m0) {
                    break;
                }
                $alpha *= 0.5;
            }
            $xn = self::project(self::axpy($alpha, $d, $x), $lo, $hi);
            $gn = $this->g($xn);
            $s = self::sub($xn, $x);
            $y = self::sub($gn, $gk);
            $sy = self::dot($s, $y);
            if ($sy > 1e-10) {
                $Bs = self::matvec($B, $s);
                $sBs = self::dot($s, $Bs);
                for ($i = 0; $i < $n; $i++) {
                    for ($jj = 0; $jj < $n; $jj++) {
                        $B[$i][$jj] = $B[$i][$jj] - $Bs[$i] * $Bs[$jj] / $sBs + $y[$i] * $y[$jj] / $sy;
                    }
                }
            }
            $done = self::norm2(self::sub($xn, $x)) < $tol * max(1.0, self::norm2($x));
            $x = $xn;
            $f = $this->f($x);
            if ($done) {
                $status = 0;
                $msg = 'Optimization terminated successfully';
                break;
            }
        }

        return new OptimizeResult($x, $f, $status === 0, $status, $msg, $it, $this->nfev, $this->njev, $this->g($x), null, 'SLSQP');
    }

    // ------------------------------------------------------------------ COBYLA (linear-approximation TR)

    /**
     * scipy.optimize.minimize(method='COBYLA'): Powell's constrained optimisation by linear approximation. A
     * simplex of n+1 points yields affine models of f and of every inequality constraint c_j>=0; each step
     * minimises the penalised linear merit within a box trust region of half-width rho, then the simplex is
     * updated and rho is reduced on rejected steps (rhobeg -> rhoend). A geometry safeguard rebuilds the
     * simplex when its edges become ill-conditioned. Reaches scipy's constrained optimum on well-posed
     * problems; the iterate sequence differs and very degenerate corners can be sensitive.
     *
     * @param list<float> $x
     * @param array<string, mixed> $o constraints: list<array{type:'ineq', fun:callable}>, rhobeg, rhoend, maxfun
     */
    private function cobyla(array $x0, array $o): OptimizeResult
    {
        $n = count($x0);
        $cons = [];
        foreach (($o['constraints'] ?? []) as $c) {
            $cons[] = $c['fun'];
        }
        $m = count($cons);
        $rhobeg = (float) ($o['rhobeg'] ?? 1.0);
        $rhoend = (float) ($o['rhoend'] ?? 1e-4);
        $maxfun = (int) ($o['maxfun'] ?? 3500);

        $evalC = function (array $x) use ($cons): array {
            $out = [];
            foreach ($cons as $c) {
                $out[] = (float) $c($x);
            }

            return $out;
        };
        $viol = static function (array $cv) use ($m): float {
            if ($m === 0) {
                return 0.0;
            }

            return max(0.0, -min($cv));
        };

        // simplex
        $sim = [$x0];
        for ($i = 0; $i < $n; $i++) {
            $e = $x0;
            $e[$i] += $rhobeg;
            $sim[] = $e;
        }
        $fv = [];
        $cv = [];
        foreach ($sim as $p) {
            $fv[] = $this->f($p);
            $cv[] = $evalC($p);
        }
        $rho = $rhobeg;
        $mu = 0.0;
        $merit = function (float $fval, array $cval) use (&$mu, $viol): float {
            return $fval + $mu * $viol($cval);
        };
        $rebuild = function (array $xb) use (&$sim, &$fv, &$cv, &$rho, $n, $evalC): void {
            $sim[0] = $xb;
            $fv[0] = $this->f($xb);
            $cv[0] = $evalC($xb);
            for ($i = 0; $i < $n; $i++) {
                $e = $xb;
                $e[$i] += $rho;
                $sim[$i + 1] = $e;
                $fv[$i + 1] = $this->f($e);
                $cv[$i + 1] = $evalC($e);
            }
        };

        while (true) {
            for ($inner = 0; $inner < 200; $inner++) {
                // order vertices by merit; best is the model base
                $order = range(0, $n);
                usort($order, fn ($a, $b) => $merit($fv[$a], $cv[$a]) <=> $merit($fv[$b], $cv[$b]));
                $b = $order[0];
                $rows = array_values(array_filter(range(0, $n), fn ($i) => $i !== $b));
                $M = [];
                foreach ($rows as $i) {
                    $M[] = self::sub($sim[$i], $sim[$b]);
                }
                // geometry safeguard: rebuild if edges degenerate (singular / mis-scaled)
                $edge = 0.0;
                foreach ($M as $row) {
                    $edge = max($edge, self::norm2($row));
                }
                $det = self::absDet($M);
                if ($edge < 0.25 * $rho || $edge > 4.0 * $rho || $det < 1e-12 * ($rho ** $n)) {
                    $rebuild($sim[$b]);
                    $order = range(0, $n);
                    usort($order, fn ($a, $c) => $merit($fv[$a], $cv[$a]) <=> $merit($fv[$c], $cv[$c]));
                    $b = $order[0];
                    $rows = array_values(array_filter(range(0, $n), fn ($i) => $i !== $b));
                    $M = [];
                    foreach ($rows as $i) {
                        $M[] = self::sub($sim[$i], $sim[$b]);
                    }
                }
                $Minv = self::pinvSmall($M);          // (n x n): columns index rows[]
                $gf = self::matTvec($Minv, array_map(fn ($i) => $fv[$i] - $fv[$b], $rows));
                $gc = [];
                for ($j = 0; $j < $m; $j++) {
                    $gc[$j] = self::matTvec($Minv, array_map(fn ($i) => $cv[$i][$j] - $cv[$b][$j], $rows));
                }
                $c0 = $cv[$b];
                if ($m > 0) {
                    $muNeed = self::norm2($gf) / max($viol($c0), $rho);
                    $mu = max(min($muNeed, 2.0 * ($mu + 1.0)), $mu);
                }
                $d = self::cobylaStep($gf, $gc, $c0, $rho, $mu, $n, $m);
                if (self::normInf($d) < 1e-3 * $rho) {
                    break;
                }
                $xnew = self::axpy(1.0, $d, $sim[$b]);
                $fn = $this->f($xnew);
                $cn = $evalC($xnew);
                // worst non-base vertex by merit
                $worst = $rows[0];
                foreach ($rows as $i) {
                    if ($merit($fv[$i], $cv[$i]) > $merit($fv[$worst], $cv[$worst])) {
                        $worst = $i;
                    }
                }
                if ($merit($fn, $cn) < $merit($fv[$worst], $cv[$worst]) - 1e-14) {
                    $sim[$worst] = $xnew;
                    $fv[$worst] = $fn;
                    $cv[$worst] = $cn;
                } else {
                    $rho *= 0.5;
                    $rebuild($sim[$b]);
                    if ($rho <= $rhoend) {
                        break;
                    }
                }
                if ($this->nfev > $maxfun) {
                    break;
                }
            }
            if ($rho <= $rhoend || $this->nfev > $maxfun) {
                break;
            }
            $rho *= 0.5;
            $order = range(0, $n);
            usort($order, fn ($a, $b) => $merit($fv[$a], $cv[$a]) <=> $merit($fv[$b], $cv[$b]));
            $rebuild($sim[$order[0]]);
        }
        $order = range(0, $n);
        usort($order, fn ($a, $b) => $merit($fv[$a], $cv[$a]) <=> $merit($fv[$b], $cv[$b]));
        $b = $order[0];
        $success = $m === 0 || $viol($cv[$b]) <= (float) ($o['catol'] ?? 2e-4);

        return new OptimizeResult($sim[$b], $fv[$b], $success, $success ? 1 : 4, $success ? 'Optimization terminated successfully.' : 'Did not converge to a feasible point.', $this->nfev, $this->nfev, 0, null, null, 'COBYLA');
    }

    /**
     * COBYLA trust-region subproblem: minimise  gf.d + mu * max_j max(0, -(c0_j + gc_j . d))  over the box
     * ||d||_inf <= rho, by projected subgradient descent with a coordinate polish (reliable at the small n
     * the constrained minimiser covers).
     *
     * @param list<float> $gf
     * @param list<list<float>> $gc
     * @param list<float> $c0
     */
    private static function cobylaStep(array $gf, array $gc, array $c0, float $rho, float $mu, int $n, int $m): array
    {
        $meritLin = static function (array $d) use ($gf, $gc, $c0, $mu, $m): float {
            $v = 0.0;
            for ($j = 0; $j < $m; $j++) {
                $v = max($v, max(0.0, -($c0[$j] + self::dot($gc[$j], $d))));
            }

            return self::dot($gf, $d) + $mu * $v;
        };
        $subgrad = static function (array $d) use ($gf, $gc, $c0, $mu, $m, $n): array {
            $g = $gf;
            $worst = -1.0;
            $wj = -1;
            for ($j = 0; $j < $m; $j++) {
                $vj = -($c0[$j] + self::dot($gc[$j], $d));
                if ($vj > $worst) {
                    $worst = $vj;
                    $wj = $j;
                }
            }
            if ($worst > 0 && $wj >= 0) {
                $g = self::axpy(-$mu, $gc[$wj], $g);
            }

            return $g;
        };
        $clip = static function (array $d) use ($rho, $n): array {
            for ($i = 0; $i < $n; $i++) {
                $d[$i] = min(max($d[$i], -$rho), $rho);
            }

            return $d;
        };
        $d = array_fill(0, $n, 0.0);
        $best = $meritLin($d);
        $bestd = $d;
        $lr = $rho;
        for ($t = 0; $t < 400; $t++) {
            $g = $subgrad($d);
            $ng = self::norm2($g);
            if ($ng < 1e-14) {
                break;
            }
            $d = $clip(self::axpy(-$lr / $ng, $g, $d));
            $mv = $meritLin($d);
            if ($mv < $best - 1e-16) {
                $best = $mv;
                $bestd = $d;
            } else {
                $lr *= 0.7;
            }
            if ($lr < 1e-6 * $rho) {
                break;
            }
        }
        $d = $bestd;
        $step = $rho;
        for ($t = 0; $t < 80; $t++) {
            $improved = false;
            for ($i = 0; $i < $n; $i++) {
                foreach ([$step, -$step] as $s) {
                    $dd = $d;
                    $dd[$i] = min(max($dd[$i] + $s, -$rho), $rho);
                    $mv = $meritLin($dd);
                    if ($mv < $best - 1e-16) {
                        $best = $mv;
                        $d = $dd;
                        $improved = true;
                    }
                }
            }
            if (! $improved) {
                $step *= 0.5;
                if ($step < 1e-4 * $rho) {
                    break;
                }
            }
        }

        return $d;
    }

    /**
     * Active-set QP: minimise 1/2 d'Bd + g'd subject to Aeq d = -beq and Cin d >= -bin and box
     * lo-x <= d <= hi-x. Used by SLSQP (and by extension) as the step subproblem.
     *
     * @param list<list<float>> $B
     * @param list<float> $g
     * @param list<list<float>> $Ceq
     * @param list<float> $beq
     * @param list<list<float>> $Cin
     * @param list<float> $bin
     * @param list<float> $lo
     * @param list<float> $hi
     * @param list<float> $x
     */
    private static function solveQp(array $B, array $g, array $Ceq, array $beq, array $Cin, array $bin, array $lo, array $hi, array $x): array
    {
        $n = count($g);
        // assemble inequalities (>= form):  row . d >= rhs
        $Crows = [];
        $brows = [];
        foreach ($Cin as $i => $row) {
            $Crows[] = $row;
            $brows[] = -$bin[$i];
        }
        for ($i = 0; $i < $n; $i++) {
            if (is_finite($lo[$i])) {
                $e = array_fill(0, $n, 0.0);
                $e[$i] = 1.0;
                $Crows[] = $e;
                $brows[] = $lo[$i] - $x[$i];
            }
            if (is_finite($hi[$i])) {
                $e = array_fill(0, $n, 0.0);
                $e[$i] = -1.0;
                $Crows[] = $e;
                $brows[] = -($hi[$i] - $x[$i]);
            }
        }
        $nbeq = count($beq);
        // equality rhs: Aeq d = -beq
        $eqRhs = array_map(static fn (float $v): float => -$v, $beq);
        $active = [];
        $d = array_fill(0, $n, 0.0);
        for ($iter = 0; $iter < 200; $iter++) {
            // rows = equalities ++ active inequalities
            $Amat = $Ceq;
            $bvec = $eqRhs;
            foreach ($active as $a) {
                $Amat[] = $Crows[$a];
                $bvec[] = $brows[$a];
            }
            $k = count($Amat);
            // build KKT [[B, -A'],[A, 0]] [d; lam] = [-g; bvec]
            $dim = $n + $k;
            $K = [];
            for ($i = 0; $i < $dim; $i++) {
                $K[$i] = array_fill(0, $dim, 0.0);
            }
            for ($i = 0; $i < $n; $i++) {
                for ($jj = 0; $jj < $n; $jj++) {
                    $K[$i][$jj] = $B[$i][$jj];
                }
                $K[$i][$i] += 1e-12;
            }
            for ($r = 0; $r < $k; $r++) {
                for ($jj = 0; $jj < $n; $jj++) {
                    $K[$jj][$n + $r] = -$Amat[$r][$jj];
                    $K[$n + $r][$jj] = $Amat[$r][$jj];
                }
            }
            $rhs = [];
            for ($i = 0; $i < $n; $i++) {
                $rhs[$i] = -$g[$i];
            }
            for ($r = 0; $r < $k; $r++) {
                $rhs[$n + $r] = $bvec[$r];
            }
            $sol = self::linSolve($K, $rhs);
            $d = array_slice($sol, 0, $n);
            $lam = array_slice($sol, $n);
            // find violated inactive inequalities
            $viol = [];
            foreach ($Crows as $i => $row) {
                if (in_array($i, $active, true)) {
                    continue;
                }
                if (self::dot($row, $d) - $brows[$i] < -1e-10) {
                    $viol[] = $i;
                }
            }
            if ($viol === []) {
                // check inequality multipliers (indices after equalities) >= 0
                $neg = -1;
                $negval = 0.0;
                for ($r = 0; $r < count($active); $r++) {
                    $lv = $lam[$nbeq + $r];
                    if ($lv < -1e-10 && $lv < $negval) {
                        $negval = $lv;
                        $neg = $r;
                    }
                }
                if ($neg < 0) {
                    return $d;
                }
                array_splice($active, $neg, 1);
            } else {
                $active[] = $viol[0];
            }
        }

        return $d;
    }

    /**
     * @param list<array{0: float|null, 1: float|null}>|null $bounds
     * @return array{0: list<float>, 1: list<float>}
     */
    private static function unpackBounds(?array $bounds, int $n): array
    {
        $lo = array_fill(0, $n, -INF);
        $hi = array_fill(0, $n, INF);
        if ($bounds !== null) {
            if (count($bounds) !== $n) {
                throw new TesseroException('minimize: bounds needs one (min, max) pair per variable.');
            }
            foreach (array_values($bounds) as $i => $pair) {
                $lo[$i] = $pair[0] ?? -INF;
                $hi[$i] = $pair[1] ?? INF;
            }
        }

        return [$lo, $hi];
    }

    /** Forward-difference gradient of a scalar constraint callable at x. */
    private static function constraintGrad(callable $c, array $x): array
    {
        $rel = sqrt(self::EPS);
        $f0 = (float) $c($x);
        $g = [];
        foreach ($x as $i => $xi) {
            $h = $rel * ($xi >= 0 ? 1.0 : -1.0) * max(1.0, abs($xi));
            $xh = $x;
            $xh[$i] = $xi + $h;
            $h = $xh[$i] - $xi;
            $g[$i] = ((float) $c($xh) - $f0) / $h;
        }

        return $g;
    }

    /** Solve A y = b for a dense square A by Gaussian elimination with partial pivoting. */
    private static function linSolve(array $A, array $b): array
    {
        $n = count($b);
        $M = [];
        for ($i = 0; $i < $n; $i++) {
            $M[$i] = array_merge($A[$i], [$b[$i]]);
        }
        for ($col = 0; $col < $n; $col++) {
            $piv = $col;
            $best = abs($M[$col][$col]);
            for ($r = $col + 1; $r < $n; $r++) {
                if (abs($M[$r][$col]) > $best) {
                    $best = abs($M[$r][$col]);
                    $piv = $r;
                }
            }
            if ($piv !== $col) {
                $tmp = $M[$col];
                $M[$col] = $M[$piv];
                $M[$piv] = $tmp;
            }
            $d = $M[$col][$col];
            if (abs($d) < 1e-300) {
                $M[$col][$col] = $d = 1e-300;
            }
            for ($r = 0; $r < $n; $r++) {
                if ($r === $col) {
                    continue;
                }
                $factor = $M[$r][$col] / $d;
                if ($factor === 0.0) {
                    continue;
                }
                for ($cc = $col; $cc <= $n; $cc++) {
                    $M[$r][$cc] -= $factor * $M[$col][$cc];
                }
            }
        }
        $y = [];
        for ($i = 0; $i < $n; $i++) {
            $y[$i] = $M[$i][$n] / $M[$i][$i];
        }

        return $y;
    }

    /** Moore-Penrose pseudo-inverse of a small (n x n) matrix via normal equations; returns n x n. */
    private static function pinvSmall(array $M): array
    {
        $n = count($M);
        // M is n x n (rows are simplex edges). pinv(M) = M^{-1} if invertible, else (M'M)^{-1} M'.
        $MtM = [];
        for ($i = 0; $i < $n; $i++) {
            $MtM[$i] = array_fill(0, $n, 0.0);
            for ($j = 0; $j < $n; $j++) {
                $s = 0.0;
                for ($r = 0; $r < $n; $r++) {
                    $s += $M[$r][$i] * $M[$r][$j];
                }
                $MtM[$i][$j] = $s + ($i === $j ? 1e-14 : 0.0);
            }
        }
        // inverse of MtM via solving against identity, then pinv = inv(MtM) M'
        $inv = [];
        for ($col = 0; $col < $n; $col++) {
            $e = array_fill(0, $n, 0.0);
            $e[$col] = 1.0;
            $colsol = self::linSolve($MtM, $e);
            for ($i = 0; $i < $n; $i++) {
                $inv[$i][$col] = $colsol[$i];
            }
        }
        // pinv (n x n): pinv[i][r] = sum_j inv[i][j] * M[r][j]
        $pinv = [];
        for ($i = 0; $i < $n; $i++) {
            $pinv[$i] = array_fill(0, $n, 0.0);
            for ($r = 0; $r < $n; $r++) {
                $s = 0.0;
                for ($j = 0; $j < $n; $j++) {
                    $s += $inv[$i][$j] * $M[$r][$j];
                }
                $pinv[$i][$r] = $s;
            }
        }

        return $pinv;
    }

    /** Compute Minv^T . v where Minv is n x n and v has length n (column-indexed by simplex rows). */
    private static function matTvec(array $Minv, array $v): array
    {
        $n = count($Minv);
        $out = array_fill(0, $n, 0.0);
        for ($i = 0; $i < $n; $i++) {
            $s = 0.0;
            for ($r = 0; $r < $n; $r++) {
                $s += $Minv[$i][$r] * $v[$r];
            }
            $out[$i] = $s;
        }

        return $out;
    }

    /** Absolute determinant of a small square matrix (Gaussian elimination). */
    private static function absDet(array $M): float
    {
        $n = count($M);
        if ($n === 0) {
            return 1.0;
        }
        $A = array_map(static fn ($row) => array_values($row), $M);
        $det = 1.0;
        for ($col = 0; $col < $n; $col++) {
            $piv = $col;
            $best = abs($A[$col][$col]);
            for ($r = $col + 1; $r < $n; $r++) {
                if (abs($A[$r][$col]) > $best) {
                    $best = abs($A[$r][$col]);
                    $piv = $r;
                }
            }
            if ($best < 1e-300) {
                return 0.0;
            }
            if ($piv !== $col) {
                $tmp = $A[$col];
                $A[$col] = $A[$piv];
                $A[$piv] = $tmp;
                $det = -$det;
            }
            $det *= $A[$col][$col];
            for ($r = $col + 1; $r < $n; $r++) {
                $factor = $A[$r][$col] / $A[$col][$col];
                for ($cc = $col; $cc < $n; $cc++) {
                    $A[$r][$cc] -= $factor * $A[$col][$cc];
                }
            }
        }

        return abs($det);
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

    private function trustRegion(array $x, array $o, string $subkind): OptimizeResult
    {
        $n = count($x);
        $hess = $o['hess'] ?? null;
        $hessp = $o['hessp'] ?? null;
        if ($this->jac === null) {
            throw new TesseroException("minimize: method '{$subkind}' requires a Jacobian (jac).");
        }
        $hvpOnly = ($subkind === 'trust-ncg' || $subkind === 'trust-krylov');
        if (! $hvpOnly && ! is_callable($hess)) {
            throw new TesseroException("minimize: method '{$subkind}' requires a callable Hessian (options['hess']).");
        }
        if ($hvpOnly && ! is_callable($hess) && ! is_callable($hessp)) {
            throw new TesseroException("minimize: method '{$subkind}' requires options['hess'] or options['hessp'].");
        }

        $eta = (float) ($o['eta'] ?? 0.15);
        $gtol = (float) ($o['gtol'] ?? 1e-4);
        $tr = (float) ($o['initial_trust_radius'] ?? 1.0);
        $maxTr = (float) ($o['max_trust_radius'] ?? 1000.0);
        $maxiter = (int) ($o['maxiter'] ?? 200 * $n);
        $subMaxiter = (int) ($o['subproblem_maxiter'] ?? 25);

        // Hessian-vector product used by the model and by the Steihaug CG solver.
        $hvp = static function (array $xp, array $p) use ($hess, $hessp): array {
            if ($hessp !== null) {
                return array_map('floatval', array_values($hessp($xp, $p)));
            }
            $H = $hess($xp);

            return self::matvec($H, $p);
        };

        // per-iteration cache for trust-exact (lambda lower bound reuse across shrinking radii)
        $exactState = ['prev' => -1.0, 'lb' => null];

        $warn = 0;
        $msg = 'Optimization terminated successfully.';
        $fx = $this->f($x);
        $gx = $this->g($x, $fx);
        $k = 0;
        while (self::norm2($gx) >= $gtol) {
            $Hx = ($hvpOnly && $hess === null) ? null : $hess($x);
            try {
                if ($subkind === 'dogleg') {
                    [$p, $hits] = self::subDogleg($gx, $Hx, $tr);
                } elseif ($subkind === 'trust-ncg') {
                    [$p, $hits] = self::subSteihaug($gx, $tr, static fn (array $v): array => $hvp($x, $v));
                } elseif ($subkind === 'trust-krylov') {
                    [$p, $hits] = self::subGltr($gx, $tr, static fn (array $v): array => $hvp($x, $v));
                } else {
                    [$p, $hits] = self::subExact($gx, $Hx, $tr, $subMaxiter, $exactState);
                }
            } catch (\RuntimeException $e) {   // non-PD Hessian / linalg failure, as scipy's warnflag 3
                $warn = 3;
                $msg = 'A linalg error occurred, such as a non-psd Hessian.';
                break;
            }

            $Hp = $hvp($x, $p);
            $predval = $fx + self::dot($gx, $p) + 0.5 * self::dot($p, $Hp);
            $xp = self::axpy(1.0, $p, $x);
            $fxp = $this->f($xp);
            $actual = $fx - $fxp;
            $pred = $fx - $predval;
            if ($pred <= 0) {
                $warn = 2;
                $msg = 'A bad approximation caused failure to predict improvement.';
                break;
            }
            $rho = $actual / $pred;
            if ($rho < 0.25) {
                $tr *= 0.25;
            } elseif ($rho > 0.75 && $hits) {
                $tr = min(2.0 * $tr, $maxTr);
            }
            if ($rho > $eta) {
                $x = $xp;
                $fx = $fxp;
                $gx = $this->g($x, $fx);
            }
            $k++;
            if (self::norm2($gx) < $gtol) {
                break;
            }
            if ($k >= $maxiter) {
                $warn = 1;
                $msg = 'Maximum number of iterations has been exceeded.';
                break;
            }
        }

        $method = match ($subkind) { 'dogleg' => 'dogleg', 'trust-ncg' => 'trust-ncg', 'trust-krylov' => 'trust-krylov', default => 'trust-exact' };

        return new OptimizeResult($x, $fx, $warn === 0, $warn, $msg, $k, $this->nfev, $this->njev, null, null, $method);
    }

    /**
     * Solve ||z + t d|| == trust_radius for the two roots t, sorted low to high
     * (scipy BaseQuadraticSubproblem.get_boundaries_intersections; the round-off-safe form).
     *
     * @param list<float> $z
     * @param list<float> $d
     * @return array{0: float, 1: float}
     */
    private static function boundaryIntersections(array $z, array $d, float $tr): array
    {
        $a = self::dot($d, $d);
        $b = 2.0 * self::dot($z, $d);
        $c = self::dot($z, $z) - $tr * $tr;
        $sqrtDisc = sqrt($b * $b - 4.0 * $a * $c);
        $aux = $b + ($b >= 0.0 ? $sqrtDisc : -$sqrtDisc);   // copysign(sqrt_disc, b)
        $ta = -$aux / (2.0 * $a);
        $tb = -2.0 * $c / $aux;

        return $ta <= $tb ? [$ta, $tb] : [$tb, $ta];
    }

    /**
     * Dogleg subproblem (scipy DoglegSubproblem.solve). Requires a positive-definite Hessian.
     *
     * @param list<float> $g
     * @param list<list<float>> $H
     * @return array{0: list<float>, 1: bool}
     */
    private static function subDogleg(array $g, array $H, float $tr): array
    {
        $n = count($g);
        [$U, $info] = self::cholUpper($H);
        if ($info !== 0) {
            throw new \RuntimeException('Hessian is not positive definite.');
        }
        $pBest = self::choSolveUpper($U, self::neg($g));   // Newton point -H^{-1} g
        if (self::norm2($pBest) < $tr) {
            return [$pBest, false];
        }
        $Bg = self::matvec($H, $g);
        $pu = self::scale(-self::dot($g, $g) / self::dot($g, $Bg), $g);   // Cauchy point
        $puNorm = self::norm2($pu);
        if ($puNorm >= $tr) {
            return [self::scale($tr / $puNorm, $pu), true];
        }
        $diff = self::sub($pBest, $pu);
        [, $tb] = self::boundaryIntersections($pu, $diff, $tr);

        return [self::axpy($tb, $diff, $pu), true];
    }

    /**
     * Steihaug conjugate-gradient subproblem (scipy CGSteihaugSubproblem.solve, Nocedal-Wright alg 7.2).
     * Only needs Hessian-vector products; the Hessian need not be positive definite.
     *
     * @param list<float> $g
     * @param callable(list<float>): list<float> $hvp
     * @return array{0: list<float>, 1: bool}
     */
    private static function subSteihaug(array $g, float $tr, callable $hvp): array
    {
        $n = count($g);
        $gMag = self::norm2($g);
        $tol = min(0.5, sqrt($gMag)) * $gMag;
        $z = array_fill(0, $n, 0.0);
        if ($gMag < $tol) {
            return [$z, false];
        }
        $r = $g;
        $d = self::neg($r);
        $mval = static fn (array $p): float => self::dot($g, $p) + 0.5 * self::dot($p, $hvp($p));
        while (true) {
            $Bd = $hvp($d);
            $dBd = self::dot($d, $Bd);
            if ($dBd <= 0) {
                [$ta, $tb] = self::boundaryIntersections($z, $d, $tr);
                $pa = self::axpy($ta, $d, $z);
                $pb = self::axpy($tb, $d, $z);

                return [$mval($pa) < $mval($pb) ? $pa : $pb, true];
            }
            $rSq = self::dot($r, $r);
            $alpha = $rSq / $dBd;
            $zNext = self::axpy($alpha, $d, $z);
            if (self::norm2($zNext) >= $tr) {
                [, $tb] = self::boundaryIntersections($z, $d, $tr);

                return [self::axpy($tb, $d, $z), true];
            }
            $rNext = self::axpy($alpha, $Bd, $r);
            $rNextSq = self::dot($rNext, $rNext);
            if (sqrt($rNextSq) < $tol) {
                return [$zNext, false];
            }
            $beta = $rNextSq / $rSq;
            $d = self::axpy($beta, $d, self::neg($rNext));
            $z = $zNext;
            $r = $rNext;
        }
    }

    /**
     * GLTR (Generalized Lanczos Trust Region) subproblem, the method used by
     * scipy.optimize.minimize(method='trust-krylov') (which wraps the _trlib GLTR
     * solver of Gould, Lucidi, Roma & Toint). Needs only Hessian-vector products.
     *
     * Build a Lanczos tridiagonalisation of H with first vector q_1 = -g/||g||; in
     * that basis the trust-region model is  1/2 h^T T h + ||g|| e_1 . h. If the
     * unconstrained (CG) solution of (interior) T h = ||g|| e_1 lies inside the
     * trust region and the model gradient g + H p is small, return it; otherwise
     * solve the tridiagonal boundary problem ||h(lam)|| = tr by a safeguarded
     * Newton iteration on the Moré-Sorensen secular equation 1/tr - 1/||h(lam)|| = 0,
     * with the lower bound lam >= max(0, -lambda_min(T)).
     *
     * Converges to the same minimiser scipy's trust-krylov reaches (unique-min
     * problems match to ~1e-5 relative). The outer trust-region loop is shared with
     * dogleg / trust-ncg / trust-exact, so only the step direction differs.
     *
     * @param list<float> $g
     * @param callable(list<float>): list<float> $hvp
     * @return array{0: list<float>, 1: bool}
     */
    private static function subGltr(array $g, float $tr, callable $hvp): array
    {
        $n = count($g);
        $gnorm = self::norm2($g);
        if ($gnorm === 0.0) {
            return [array_fill(0, $n, 0.0), false];
        }
        $maxiter = 200;
        $cgTol = min(1e-11, $gnorm) * $gnorm;

        /** @var list<list<float>> $Q */
        $Q = [self::scale(-1.0 / $gnorm, $g)];   // q_1 = -g/||g||
        /** @var list<float> $alpha */
        $alpha = [];
        /** @var list<float> $beta */
        $beta = [];

        $lim = min($maxiter, $n);
        for ($j = 0; $j < $lim; $j++) {
            $w = $hvp($Q[$j]);
            $a = self::dot($Q[$j], $w);
            $alpha[] = $a;
            $w = self::axpy(-$a, $Q[$j], $w);
            if ($j > 0) {
                $w = self::axpy(-$beta[$j - 1], $Q[$j - 1], $w);
            }
            // full reorthogonalisation (short Lanczos recurrences lose orthogonality)
            foreach ($Q as $qi) {
                $w = self::axpy(-self::dot($qi, $w), $qi, $w);
            }
            $b = self::norm2($w);

            // interior (lam = 0) solution attempt: (T) h = ||g|| e_1
            if (self::tridiagIsPd($alpha, $beta, 0.0)) {
                $m = count($alpha);
                $rhs = array_fill(0, $m, 0.0);
                $rhs[0] = $gnorm;
                $h = self::tridiagSolveShifted($alpha, $beta, 0.0, $rhs);
                $p = array_fill(0, $n, 0.0);
                foreach ($h as $i => $hi) {
                    $p = self::axpy($hi, $Q[$i], $p);
                }
                if (self::norm2($p) <= $tr * (1.0 + 1e-12)) {
                    $grad = self::axpy(1.0, $g, $hvp($p));   // g + H p
                    if (self::norm2($grad) < $cgTol) {
                        return [$p, false];
                    }
                }
            }

            if ($b < 1e-14) {
                break;
            }
            $beta[] = $b;
            $Q[] = self::scale(1.0 / $b, $w);
        }

        // Boundary solution: min 1/2 h^T T h + ||g|| e_1 . h  s.t. ||h|| = tr.
        $k = count($alpha);
        $rhs = array_fill(0, $k, 0.0);
        $rhs[0] = $gnorm;
        $lamMin = self::tridiagMinEig($alpha, $beta);
        $lamLower = max(0.0, -$lamMin);
        $lam = ($lamLower > 0.0) ? $lamLower + max(1.0, $lamLower) * 1e-3 : 0.0;

        for ($iter = 0; $iter < 100; $iter++) {
            while (! self::tridiagIsPd($alpha, $beta, $lam) && $lam < 1e18) {
                $lam = $lam * 2.0 + 1.0;
            }
            $h = self::tridiagSolveShifted($alpha, $beta, $lam, $rhs);
            $hn = self::norm2($h);
            if ($hn === 0.0) {
                $lam = $lam * 2.0 + 1.0;
                continue;
            }
            $phi = 1.0 / $tr - 1.0 / $hn;
            if (abs($phi) < 1e-12) {
                break;
            }
            $qv = self::tridiagSolveShifted($alpha, $beta, $lam, $h);
            $hq = self::dot($h, $qv);
            if ($hq <= 0.0) {
                $lam = $lam * 2.0 + 1.0;
                continue;
            }
            $denom = $hq / ($hn ** 3);
            $lamNew = $lam - $phi / $denom;
            if ($lamNew < $lamLower) {
                $lamNew = ($lam > $lamLower) ? ($lam + $lamLower) / 2.0 : $lamLower + 1e-8;
            }
            if (abs($lamNew - $lam) < 1e-14) {
                $lam = $lamNew;
                break;
            }
            $lam = $lamNew;
        }

        $h = self::tridiagSolveShifted($alpha, $beta, $lam, $rhs);
        $hn = self::norm2($h);
        if ($hn > 0.0) {
            $h = self::scale($tr / $hn, $h);   // project onto the boundary (guard)
        }
        $p = array_fill(0, $n, 0.0);
        foreach ($h as $i => $hi) {
            $p = self::axpy($hi, $Q[$i], $p);
        }

        return [$p, true];
    }

    /**
     * Thomas algorithm for (T + lam*I) h = rhs, with T symmetric tridiagonal
     * (diagonal $alpha, off-diagonal $beta of length n-1).
     *
     * @param list<float> $alpha
     * @param list<float> $beta
     * @param list<float> $rhs
     * @return list<float>
     */
    private static function tridiagSolveShifted(array $alpha, array $beta, float $lam, array $rhs): array
    {
        $n = count($alpha);
        $d = [];
        for ($i = 0; $i < $n; $i++) {
            $d[$i] = $alpha[$i] + $lam;
        }
        $cp = array_fill(0, $n, 0.0);
        $dp = array_fill(0, $n, 0.0);
        $cp[0] = ($n > 1) ? ($beta[0] / $d[0]) : 0.0;
        $dp[0] = $rhs[0] / $d[0];
        for ($i = 1; $i < $n; $i++) {
            $denom = $d[$i] - $beta[$i - 1] * $cp[$i - 1];
            if ($i < $n - 1) {
                $cp[$i] = $beta[$i] / $denom;
            }
            $dp[$i] = ($rhs[$i] - $beta[$i - 1] * $dp[$i - 1]) / $denom;
        }
        $h = array_fill(0, $n, 0.0);
        $h[$n - 1] = $dp[$n - 1];
        for ($i = $n - 2; $i >= 0; $i--) {
            $h[$i] = $dp[$i] - $cp[$i] * $h[$i + 1];
        }

        return $h;
    }

    /**
     * Positive-definiteness test of T + lam*I via the LDL^T pivots (all > 0).
     *
     * @param list<float> $alpha
     * @param list<float> $beta
     */
    private static function tridiagIsPd(array $alpha, array $beta, float $lam): bool
    {
        $n = count($alpha);
        $dp = $alpha[0] + $lam;
        if ($dp <= 0.0) {
            return false;
        }
        for ($i = 1; $i < $n; $i++) {
            $d = ($alpha[$i] + $lam) - ($beta[$i - 1] ** 2) / $dp;
            if ($d <= 0.0) {
                return false;
            }
            $dp = $d;
        }

        return true;
    }

    /**
     * Smallest eigenvalue of a symmetric tridiagonal matrix (diagonal $alpha,
     * off-diagonal $beta) by bisection on the Sturm sequence (negative-pivot count
     * of T - x*I), bracketed by the Gershgorin interval. Used for the Moré-Sorensen
     * lower bound lam >= -lambda_min.
     *
     * @param list<float> $alpha
     * @param list<float> $beta
     */
    private static function tridiagMinEig(array $alpha, array $beta): float
    {
        $n = count($alpha);
        if ($n === 1) {
            return $alpha[0];
        }
        $lo = INF;
        $hi = -INF;
        for ($i = 0; $i < $n; $i++) {
            $r = 0.0;
            if ($i > 0) {
                $r += abs($beta[$i - 1]);
            }
            if ($i < $n - 1) {
                $r += abs($beta[$i]);
            }
            $lo = min($lo, $alpha[$i] - $r);
            $hi = max($hi, $alpha[$i] + $r);
        }
        $sturm = static function (float $x) use ($alpha, $beta, $n): int {
            $cnt = 0;
            $d = $alpha[0] - $x;
            if ($d < 0.0) {
                $cnt++;
            }
            for ($i = 1; $i < $n; $i++) {
                if ($d === 0.0) {
                    $d = 1e-300;
                }
                $d = ($alpha[$i] - $x) - ($beta[$i - 1] ** 2) / $d;
                if ($d < 0.0) {
                    $cnt++;
                }
            }
            return $cnt;
        };
        $a = $lo - 1.0;
        $b = $hi + 1.0;
        for ($it = 0; $it < 200; $it++) {
            $mid = 0.5 * ($a + $b);
            if ($sturm($mid) >= 1) {
                $b = $mid;
            } else {
                $a = $mid;
            }
            if ($b - $a < 1e-14 * (1.0 + abs($b))) {
                break;
            }
        }

        return 0.5 * ($a + $b);
    }

    /**
     * Nearly-exact More-Sorensen subproblem (scipy IterativeSubproblem.solve). Requires the Hessian.
     *
     * @param list<float> $g
     * @param list<list<float>> $H
     * @param array{prev: float, lb: float|null} $state  carried across calls for lambda-lower-bound reuse
     * @return array{0: list<float>, 1: bool}
     */
    private static function subExact(array $g, array $H, float $tr, int $maxiter, array &$state): array
    {
        $n = count($H);
        $gMag = self::norm2($g);
        $uc = 0.01;
        $kEasy = 0.1;
        $kHard = 0.2;

        [$glb, $gub] = self::gershgorin($H);
        $hInf = self::normInfMat($H);
        $hFro = self::normFro($H);
        $closeToZero = $n * self::EPS * $hInf;

        $diag = [];
        for ($i = 0; $i < $n; $i++) {
            $diag[] = $H[$i][$i];
        }
        $minDiag = min($diag);

        $lub = max(0.0, $gMag / $tr + min(-$glb, $hFro, $hInf));
        $llb = max(0.0, -$minDiag, $gMag / $tr - min($gub, $hFro, $hInf));
        if ($tr < $state['prev'] && $state['lb'] !== null) {
            $llb = max($state['lb'], $llb);
        }
        $lam = ($llb == 0.0) ? 0.0 : max(sqrt($llb * $lub), $llb + $uc * ($lub - $llb));

        $hits = true;
        $alreadyFactorized = false;
        $it = 0;
        $p = array_fill(0, $n, 0.0);
        $Hl = [];
        $U = [];
        $info = 0;
        while ($it < $maxiter) {
            if ($alreadyFactorized) {
                $alreadyFactorized = false;
            } else {
                $Hl = self::addDiag($H, $lam);
                [$U, $info] = self::cholUpper($Hl);
            }
            $it++;

            if ($info === 0 && $gMag > $closeToZero) {
                $p = self::choSolveUpper($U, self::neg($g));
                $pNorm = self::norm2($p);
                if ($pNorm <= $tr && $lam == 0.0) {
                    $hits = false;
                    break;
                }
                $w = self::solveTriangularUpperTrans($U, $p);   // U^T w = p
                $wNorm = self::norm2($w);
                $deltaLam = ($pNorm / $wNorm) ** 2 * ($pNorm - $tr) / $tr;
                $lamNew = $lam + $deltaLam;

                if ($pNorm < $tr) {   // inside the boundary (hard case handling)
                    [$sMin, $zMin] = self::smallestSingular($U);
                    [$ta, $tb] = self::boundaryIntersections($p, $zMin, $tr);
                    $stepLen = (abs($ta) <= abs($tb)) ? $ta : $tb;
                    $quadTerm = self::dot($p, self::matvec($H, $p));
                    $relErr = ($stepLen ** 2 * $sMin ** 2) / ($quadTerm + $lam * $tr ** 2);
                    if ($relErr <= $kHard) {
                        $p = self::axpy($stepLen, $zMin, $p);
                        break;
                    }
                    $lub = $lam;
                    $llb = max($llb, $lam - $sMin ** 2);
                    $Hl = self::addDiag($H, $lamNew);
                    [$c, $info] = self::cholUpper($Hl);
                    if ($info === 0) {
                        $lam = $lamNew;
                        $alreadyFactorized = true;
                        $U = $c;
                    } else {
                        $llb = max($llb, $lamNew);
                        $lam = max(sqrt(abs($llb * $lub)), $llb + $uc * ($lub - $llb));
                    }
                } else {   // outside the boundary
                    $relErr = abs($pNorm - $tr) / $tr;
                    if ($relErr <= $kEasy) {
                        break;
                    }
                    $llb = $lam;
                    $lam = $lamNew;
                }
            } elseif ($info === 0 && $gMag <= $closeToZero) {
                if ($lam == 0.0) {
                    $p = array_fill(0, $n, 0.0);
                    $hits = false;
                    break;
                }
                [$sMin, $zMin] = self::smallestSingular($U);
                $p = self::scale($tr, $zMin);
                if ($tr ** 2 * $sMin ** 2 <= $kHard * $lam * $tr ** 2) {
                    break;
                }
                $lub = $lam;
                $llb = max($llb, $lam - $sMin ** 2);
                $lam = max(sqrt($llb * $lub), $llb + $uc * ($lub - $llb));
            } else {   // unsuccessful factorization at leading minor $info
                [$delta, $v] = self::singularLeadingSubmatrix($Hl, $U, $info);
                $vNorm = self::norm2($v);
                $llb = max($llb, $lam + $delta / ($vNorm ** 2));
                $lam = max(sqrt(abs($llb * $lub)), $llb + $uc * ($lub - $llb));
            }
        }

        $state['lb'] = $llb;
        $state['prev'] = $tr;

        return [$p, $hits];
    }

    /**
     * Upper-triangular Cholesky H = U^T U with LAPACK potrf(clean=True) semantics: returns [U, info].
     * On success info === 0; on a non-PD leading minor k (1-based), info === k and the leading
     * (k-1)x(k-1) block of U is a valid factor (upper entries above it zeroed, as clean=True).
     *
     * @param list<list<float>> $A
     * @return array{0: list<list<float>>, 1: int}
     */
    private static function cholUpper(array $A): array
    {
        $n = count($A);
        $U = [];
        for ($i = 0; $i < $n; $i++) {
            $U[] = array_fill(0, $n, 0.0);
        }
        for ($j = 0; $j < $n; $j++) {
            $s = $A[$j][$j];
            for ($k = 0; $k < $j; $k++) {
                $s -= $U[$k][$j] * $U[$k][$j];
            }
            if ($s <= 0.0) {
                return [$U, $j + 1];   // 1-based leading minor that failed
            }
            $ujj = sqrt($s);
            $U[$j][$j] = $ujj;
            for ($i = $j + 1; $i < $n; $i++) {
                $t = $A[$j][$i];
                for ($k = 0; $k < $j; $k++) {
                    $t -= $U[$k][$j] * $U[$k][$i];
                }
                $U[$j][$i] = $t / $ujj;
            }
        }

        return [$U, 0];
    }

    /**
     * Solve H x = b given the upper Cholesky factor U (H = U^T U): forward solve U^T y = b, back solve U x = y.
     *
     * @param list<list<float>> $U
     * @param list<float> $b
     * @return list<float>
     */
    private static function choSolveUpper(array $U, array $b): array
    {
        $n = count($U);
        $y = array_fill(0, $n, 0.0);
        for ($i = 0; $i < $n; $i++) {   // forward: U^T y = b
            $s = $b[$i];
            for ($k = 0; $k < $i; $k++) {
                $s -= $U[$k][$i] * $y[$k];
            }
            $y[$i] = $s / $U[$i][$i];
        }
        $x = array_fill(0, $n, 0.0);
        for ($i = $n - 1; $i >= 0; $i--) {   // back: U x = y
            $s = $y[$i];
            for ($k = $i + 1; $k < $n; $k++) {
                $s -= $U[$i][$k] * $x[$k];
            }
            $x[$i] = $s / $U[$i][$i];
        }

        return $x;
    }

    /**
     * Solve U^T w = p for w (scipy solve_triangular(U, p, trans='T')).
     *
     * @param list<list<float>> $U
     * @param list<float> $p
     * @return list<float>
     */
    private static function solveTriangularUpperTrans(array $U, array $p): array
    {
        $n = count($U);
        $w = array_fill(0, $n, 0.0);
        for ($i = 0; $i < $n; $i++) {
            $s = $p[$i];
            for ($k = 0; $k < $i; $k++) {
                $s -= $U[$k][$i] * $w[$k];
            }
            $w[$i] = $s / $U[$i][$i];
        }

        return $w;
    }

    /**
     * Solve U x = b for an upper-triangular U (scipy solve_triangular, no transpose).
     *
     * @param list<list<float>> $U
     * @param list<float> $b
     * @return list<float>
     */
    private static function solveTriangularUpper(array $U, array $b): array
    {
        $n = count($U);
        $x = array_fill(0, $n, 0.0);
        for ($i = $n - 1; $i >= 0; $i--) {
            $s = $b[$i];
            for ($k = $i + 1; $k < $n; $k++) {
                $s -= $U[$i][$k] * $x[$k];
            }
            $x[$i] = $s / $U[$i][$i];
        }

        return $x;
    }

    /**
     * Estimate the smallest singular value of upper-triangular U and its right singular vector
     * (scipy estimate_smallest_singular_value; Cline et al. 1979 condition estimator).
     *
     * @param list<list<float>> $U
     * @return array{0: float, 1: list<float>}
     */
    private static function smallestSingular(array $U): array
    {
        $n = count($U);
        $p = array_fill(0, $n, 0.0);
        $w = array_fill(0, $n, 0.0);
        for ($k = 0; $k < $n; $k++) {
            $ukk = $U[$k][$k];   // U.T[k,k] == U[k,k]
            $wp = (1.0 - $p[$k]) / $ukk;
            $wm = (-1.0 - $p[$k]) / $ukk;
            $ppSum = 0.0;
            $pmSum = 0.0;
            $pp = [];
            $pm = [];
            for ($j = $k + 1; $j < $n; $j++) {
                $ujk = $U[$k][$j];   // U.T[j,k] == U[k,j]
                $ppv = $p[$j] + $ujk * $wp;
                $pmv = $p[$j] + $ujk * $wm;
                $pp[$j] = $ppv;
                $pm[$j] = $pmv;
                $ppSum += abs($ppv);
                $pmSum += abs($pmv);
            }
            if (abs($wp) + $ppSum >= abs($wm) + $pmSum) {
                $w[$k] = $wp;
                for ($j = $k + 1; $j < $n; $j++) {
                    $p[$j] = $pp[$j];
                }
            } else {
                $w[$k] = $wm;
                for ($j = $k + 1; $j < $n; $j++) {
                    $p[$j] = $pm[$j];
                }
            }
        }
        $v = self::solveTriangularUpper($U, $w);
        $vNorm = self::norm2($v);
        $wNorm = self::norm2($w);

        return [$wNorm / $vNorm, self::scale(1.0 / $vNorm, $v)];
    }

    /**
     * The term that makes the leading k-by-k submatrix of A singular (scipy singular_leading_submatrix).
     *
     * @param list<list<float>> $A
     * @param list<list<float>> $U
     * @return array{0: float, 1: list<float>}
     */
    private static function singularLeadingSubmatrix(array $A, array $U, int $k): array
    {
        $n = count($A);
        $delta = -$A[$k - 1][$k - 1];
        for ($i = 0; $i < $k - 1; $i++) {
            $delta += $U[$i][$k - 1] ** 2;
        }
        $v = array_fill(0, $n, 0.0);
        $v[$k - 1] = 1.0;
        if ($k !== 1) {
            // solve U[:k-1,:k-1] x = -U[:k-1,k-1]
            $sub = [];
            $rhs = [];
            for ($i = 0; $i < $k - 1; $i++) {
                $row = [];
                for ($j = 0; $j < $k - 1; $j++) {
                    $row[] = $U[$i][$j];
                }
                $sub[] = $row;
                $rhs[] = -$U[$i][$k - 1];
            }
            $x = self::solveTriangularUpper($sub, $rhs);
            for ($i = 0; $i < $k - 1; $i++) {
                $v[$i] = $x[$i];
            }
        }

        return [$delta, $v];
    }

    /**
     * Gregoshgorin bounds [lb, ub] for the eigenvalues of symmetric H (scipy gershgorin_bounds).
     *
     * @param list<list<float>> $H
     * @return array{0: float, 1: float}
     */
    private static function gershgorin(array $H): array
    {
        $n = count($H);
        $lb = INF;
        $ub = -INF;
        for ($i = 0; $i < $n; $i++) {
            $d = $H[$i][$i];
            $rowSum = 0.0;
            for ($j = 0; $j < $n; $j++) {
                $rowSum += abs($H[$i][$j]);
            }
            $lb = min($lb, $d + abs($d) - $rowSum);
            $ub = max($ub, $d - abs($d) + $rowSum);
        }

        return [$lb, $ub];
    }

    /** Induced infinity-norm of a matrix (max absolute row sum), == scipy norm(H, inf). */
    private static function normInfMat(array $H): float
    {
        $m = 0.0;
        foreach ($H as $row) {
            $s = 0.0;
            foreach ($row as $v) {
                $s += abs($v);
            }
            $m = max($m, $s);
        }

        return $m;
    }

    /** Frobenius norm of a matrix (scipy norm(H, 'fro')). */
    private static function normFro(array $H): float
    {
        $s = 0.0;
        foreach ($H as $row) {
            foreach ($row as $v) {
                $s += $v * $v;
            }
        }

        return sqrt($s);
    }

    /**
     * Return H + lambda * I.
     *
     * @param list<list<float>> $H
     * @return list<list<float>>
     */
    private static function addDiag(array $H, float $lam): array
    {
        $out = $H;
        $n = count($H);
        for ($i = 0; $i < $n; $i++) {
            $out[$i][$i] += $lam;
        }

        return $out;
    }
}
