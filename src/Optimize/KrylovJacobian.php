<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize._nonlin.KrylovJacobian: a matrix-free (Jacobian-free) Newton-Krylov
 * approximation. The action of the Jacobian J(x0) on a vector v is obtained by a
 * first-order finite difference of the residual,
 *
 *     J v ~= (F(x0 + omega*v/||v||) - F(x0)) / (omega/||v||),
 *
 * with omega = rdiff * max(1,||x0||) / max(1,||F0||) and rdiff = sqrt(eps). The
 * inner Newton step solves J dx = -F approximately with restarted GMRES
 * (scipy's default inner solver, method='gmres', inner_maxiter=20), using the
 * inexact-Newton forcing tolerance passed down from the driver.
 *
 * This reproduces scipy.optimize.newton_krylov(F, x0): the same outer
 * inexact-Newton driver ({@see NonlinSolve::nonlinSolve}) with this Jacobian.
 * Real-valued systems only. Backend-neutral pure PHP, so both Tessero backends
 * produce identical results.
 *
 * @internal
 */
final class KrylovJacobian implements JacobianApprox
{
    private const EPS = 2.220446049250313e-16;

    /** @var callable(list<float>): list<float> */
    private $func;

    /** @var list<float> */
    private array $x0 = [];

    /** @var list<float> */
    private array $f0 = [];

    private float $rdiff;

    private float $omega = 1.0;

    private int $innerMaxiter;

    public function __construct(int $innerMaxiter = 20)
    {
        $this->innerMaxiter = $innerMaxiter;
        $this->rdiff = sqrt(self::EPS);
    }

    public function setup(array $x, array $F, callable $func): void
    {
        $this->func = $func;
        $this->x0 = $x;
        $this->f0 = $F;
        $this->computeOmega();
    }

    public function update(array $x, array $f): void
    {
        $this->x0 = $x;
        $this->f0 = $f;
        $this->computeOmega();
    }

    /**
     * Solve J dx = rhs with restarted GMRES to the inexact-Newton tolerance `tol`
     * (scipy passes tol = min(eta, eta*||F||) from the driver as the GMRES rtol).
     *
     * @param list<float> $rhs
     * @return list<float>
     */
    public function solve(array $f, float $tol = 0.0): array
    {
        return $this->gmres($f, $tol);
    }

    /** scipy _NonlinKrylovJacobian._update_diff_step. */
    private function computeOmega(): void
    {
        $mx = self::maxAbs($this->x0);
        $mf = self::maxAbs($this->f0);
        $this->omega = $this->rdiff * max(1.0, $mx) / max(1.0, $mf);
    }

    /**
     * Finite-difference directional derivative J . v.
     *
     * @param list<float> $v
     * @return list<float>
     */
    private function matvec(array $v): array
    {
        $nv = NonlinSolve::n2($v);
        if ($nv === 0.0) {
            return array_fill(0, count($v), 0.0);
        }
        $sc = $this->omega / $nv;
        $xp = [];
        foreach ($this->x0 as $i => $xi) {
            $xp[] = $xi + $sc * $v[$i];
        }
        $fp = ($this->func)($xp);
        $inv = 1.0 / $sc;
        $out = [];
        foreach ($fp as $i => $fpi) {
            $out[] = ($fpi - $this->f0[$i]) * $inv;
        }
        return $out;
    }

    /**
     * One restart cycle of GMRES(m) solving J dx = b to relative tolerance
     * `rtol` (atol = 0), mirroring scipy.sparse.linalg.gmres with maxiter=1 and
     * restart=inner_maxiter, which is what newton_krylov requests for the inner
     * solve. Right-hand side is the current residual; the returned step is the
     * approximate Newton direction before the outer sign flip.
     *
     * @param list<float> $b
     * @return list<float>
     */
    private function gmres(array $b, float $rtol): array
    {
        $n = count($b);
        $bnrm = NonlinSolve::n2($b);
        if ($bnrm === 0.0) {
            return array_fill(0, $n, 0.0);
        }
        $m = $this->innerMaxiter;
        $r = $b;
        $rnrm = NonlinSolve::n2($r);
        $tol = $rtol * $bnrm;

        /** @var list<list<float>> $Q */
        $Q = [self::scale($r, 1.0 / $rnrm)];
        /** @var list<list<float>> $H */
        $H = [];
        for ($i = 0; $i <= $m; $i++) {
            $H[$i] = array_fill(0, $m, 0.0);
        }
        $cs = array_fill(0, $m + 1, 0.0);
        $sn = array_fill(0, $m + 1, 0.0);
        $g = array_fill(0, $m + 1, 0.0);
        $g[0] = $rnrm;
        $kfinal = 0;

        for ($k = 0; $k < $m; $k++) {
            $w = $this->matvec($Q[$k]);
            // Modified Gram-Schmidt (Arnoldi).
            for ($i = 0; $i <= $k; $i++) {
                $H[$i][$k] = NonlinSolve::dot($Q[$i], $w);
                $w = self::axpy(-$H[$i][$k], $Q[$i], $w);
            }
            $H[$k + 1][$k] = NonlinSolve::n2($w);
            if ($H[$k + 1][$k] > 1e-300) {
                $Q[] = self::scale($w, 1.0 / $H[$k + 1][$k]);
            } else {
                $Q[] = array_fill(0, $n, 0.0);
            }
            // Apply previous Givens rotations to the new column.
            for ($i = 0; $i < $k; $i++) {
                $t = $cs[$i] * $H[$i][$k] + $sn[$i] * $H[$i + 1][$k];
                $H[$i + 1][$k] = -$sn[$i] * $H[$i][$k] + $cs[$i] * $H[$i + 1][$k];
                $H[$i][$k] = $t;
            }
            $denom = hypot($H[$k][$k], $H[$k + 1][$k]);
            if ($denom === 0.0) {
                $denom = 1e-300;
            }
            $cs[$k] = $H[$k][$k] / $denom;
            $sn[$k] = $H[$k + 1][$k] / $denom;
            $H[$k][$k] = $cs[$k] * $H[$k][$k] + $sn[$k] * $H[$k + 1][$k];
            $H[$k + 1][$k] = 0.0;
            $g[$k + 1] = -$sn[$k] * $g[$k];
            $g[$k] = $cs[$k] * $g[$k];
            $kfinal = $k + 1;
            if (abs($g[$k + 1]) < $tol) {
                break;
            }
        }

        // Back-substitution for the least-squares coefficients.
        $y = array_fill(0, $kfinal, 0.0);
        for ($i = $kfinal - 1; $i >= 0; $i--) {
            $s = $g[$i];
            for ($j = $i + 1; $j < $kfinal; $j++) {
                $s -= $H[$i][$j] * $y[$j];
            }
            $y[$i] = $s / $H[$i][$i];
        }
        $x = array_fill(0, $n, 0.0);
        for ($i = 0; $i < $kfinal; $i++) {
            $x = self::axpy($y[$i], $Q[$i], $x);
        }
        return $x;
    }

    /** @param list<float> $v */
    private static function maxAbs(array $v): float
    {
        $m = 0.0;
        foreach ($v as $x) {
            $a = abs($x);
            if ($a > $m) {
                $m = $a;
            }
        }
        return $m;
    }

    /** @param list<float> $v @return list<float> */
    private static function scale(array $v, float $s): array
    {
        $o = [];
        foreach ($v as $x) {
            $o[] = $x * $s;
        }
        return $o;
    }

    /** y <- a*x + y. @param list<float> $x @param list<float> $y @return list<float> */
    private static function axpy(float $a, array $x, array $y): array
    {
        $o = [];
        foreach ($x as $i => $xi) {
            $o[] = $a * $xi + $y[$i];
        }
        return $o;
    }
}
