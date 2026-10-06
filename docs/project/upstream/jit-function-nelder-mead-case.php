<?php
/*
 * Included by jit-function-nelder-mead-repro.php: the miscompile only shows when this code is in an included file.
 * PHP 8.4.21 function-level JIT miscompile. Minimize::nelderMead() as it was
 * before the workaround (whole loop in one function). With
 *   php -d opcache.enable_cli=1 -d opcache.jit=function -d opcache.jit_buffer_size=128M jit-function-nelder-mead-repro.php
 * the JIT prints "ENTRY ... live var ..." and the iteration counter becomes garbage
 * (e.g. nit=139959300736489, x unchanged). JIT off and tracing: nit=141, x[0]=0.99910115...
 */
declare(strict_types=1);

namespace TesseroRepro;


/** scipy.optimize.OptimizeResult */
final class OptimizeResult implements \JsonSerializable
{
    /**
     * @param list<float> $x
     * @param list<float>|null $jac
     * @param list<list<float>>|null $hessInv
     */
    public function __construct(
        public readonly array $x,
        public readonly float $fun,
        public readonly bool $success,
        public readonly int $status,
        public readonly string $message,
        public readonly int $nit,
        public readonly int $nfev,
        public readonly int $njev = 0,
        public readonly ?array $jac = null,
        public readonly ?array $hessInv = null,
        public readonly string $method = '',
    ) {
    }

    public function jsonSerialize(): array
    {
        return array_filter(get_object_vars($this), static fn ($v): bool => $v !== null);
    }
}


use RuntimeException as TesseroException;

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

    private int $nfev = 0;

    private int $njev = 0;

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
        [$rho, $chi, $psi, $sigma] = $adaptive
            ? [1.0, 1 + 2 / $n, 0.75 - 1 / (2 * $n), 1 - 1 / $n]
            : [1.0, 2.0, 0.5, 0.5];
        $sim = [$x0];
        for ($k = 0; $k < $n; $k++) {
            $y = $x0;
            $y[$k] = $y[$k] != 0.0 ? (1 + 0.05) * $y[$k] : 0.00025;
            $sim[] = $y;
        }
        $fsim = array_map(fn (array $v): float => $this->f($v), $sim);
        self::sortSimplex($sim, $fsim);
        $iterations = 1;
        while ($this->nfev < $maxfev && $iterations < $maxiter) {
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
            $iterations++;
            self::sortSimplex($sim, $fsim);
        }
        $status = 0;
        $msg = 'Optimization terminated successfully.';
        if ($this->nfev >= $maxfev) {
            $status = 1;
            $msg = 'Maximum number of function evaluations has been exceeded.';
        } elseif ($iterations >= $maxiter) {
            $status = 2;
            $msg = 'Maximum number of iterations has been exceeded.';
        }

        return new OptimizeResult($sim[0], $fsim[0], $status === 0, $status, $msg, $iterations, $this->nfev, 0, null, null, 'Nelder-Mead');
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

$rosen = static function (array $x): float { $s = 0.0; for ($i = 0; $i < count($x) - 1; $i++) { $s += 100.0 * ($x[$i + 1] - $x[$i] ** 2.0) ** 2.0 + (1 - $x[$i]) ** 2.0; } return $s; };
$r = Minimize::minimize($rosen, [1.3, 0.7, 0.8, 1.9, 1.2], method: 'Nelder-Mead');
printf("x[0]=%.17g nit=%d nfev=%d (%s)\n", $r->x[0], $r->nit, $r->nfev, $r->message);
