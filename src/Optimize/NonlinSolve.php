<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * Pure-PHP port of scipy.optimize._nonlin.nonlin_solve: the inexact-Newton
 * driver with an Armijo backtracking line search, plus the four quasi-Newton
 * Jacobian approximations exposed as scipy.optimize.broyden1 / broyden2 /
 * anderson / diagbroyden.
 *
 * These solvers are deterministic (no RNG). The iterate sequence matches scipy
 * bit-for-bit because this reproduces exactly:
 *   - TerminationCondition (max-norm convergence check, f_tol = eps**(1/3)),
 *   - the 2-norm used for Fx_norm and the line search merit function,
 *   - LowRankMatrix (alpha*I + sum c_n d_n^T) with 'restart' rank reduction
 *     and max_rank = +inf (so it never actually reduces), for Broyden,
 *   - Anderson mixing (w0 = 0.01, M = 5) and diagonal-Broyden updates, and
 *   - scalar_search_armijo (quadratic then cubic interpolation).
 *
 * Real-valued systems only (scipy's complex path is not covered). Backend-neutral
 * pure PHP (no NDArray/BLAS), so both Tessero backends produce identical results.
 *
 * Each public method mirrors scipy.optimize.<name>(F, xin): F is a PHP callable
 * taking float[] x and returning the residual float[]; the converged root is
 * returned as float[]. On failure a {@see NoConvergence} exception is thrown
 * (carrying the best iterate on ->x), matching scipy.
 */
final class NonlinSolve
{
    private const EPS = 2.220446049250313e-16;

    /** f_tol default: np.finfo(float64).eps ** (1/3). */
    private static function fTol(): float
    {
        return self::EPS ** (1.0 / 3.0);
    }

    /** scipy.optimize.broyden1 ("good" Broyden, low-rank inverse-Jacobian update). */
    public static function broyden1(callable $F, array $xin, ?int $maxiter = null): array
    {
        return self::nonlinSolve($F, $xin, 'broyden1', $maxiter);
    }

    /** scipy.optimize.broyden2 ("bad" Broyden). */
    public static function broyden2(callable $F, array $xin, ?int $maxiter = null): array
    {
        return self::nonlinSolve($F, $xin, 'broyden2', $maxiter);
    }

    /** scipy.optimize.anderson (extended Anderson mixing). */
    public static function anderson(callable $F, array $xin, ?int $maxiter = null): array
    {
        return self::nonlinSolve($F, $xin, 'anderson', $maxiter);
    }

    /** scipy.optimize.diagbroyden (diagonal Broyden approximation). */
    public static function diagbroyden(callable $F, array $xin, ?int $maxiter = null): array
    {
        return self::nonlinSolve($F, $xin, 'diagbroyden', $maxiter);
    }

    /** scipy.optimize.linearmixing: root via a constant scalar (-alpha) Jacobian approximation. */
    public static function linearmixing(callable $F, array $xin, ?int $maxiter = null): array
    {
        return self::nonlinSolve($F, $xin, 'linearmixing', $maxiter);
    }

    /** scipy.optimize.excitingmixing: root via a tuned diagonal Jacobian approximation. */
    public static function excitingmixing(callable $F, array $xin, ?int $maxiter = null): array
    {
        return self::nonlinSolve($F, $xin, 'excitingmixing', $maxiter);
    }

    /**
     * scipy.optimize.newton_krylov: Jacobian-free (matrix-free) inexact-Newton with a
     * restarted-GMRES inner solve ({@see KrylovJacobian}). The Jacobian's action on a
     * vector is a finite difference of the residual, so no analytic/explicit Jacobian is
     * needed. Converges to the same root scipy's newton_krylov reaches (unique-root
     * systems match to ~1e-6 relative; the inner forcing tolerance follows the
     * inexact-Newton schedule, so the iterate path is numerically but not bit-for-bit
     * identical to scipy's gmres). Real-valued systems only.
     */
    public static function newtonKrylov(callable $F, array $xin, ?int $maxiter = null): array
    {
        return self::nonlinSolve($F, $xin, 'krylov', $maxiter);
    }

    /**
     * nonlin_solve driver (line_search='armijo', the scipy default).
     *
     * @param callable(list<float>): list<float> $F
     * @param list<float> $xin
     * @return list<float>
     */
    public static function nonlinSolve(callable $F, array $xin, string $jacobian, ?int $maxiter = null): array
    {
        $fTol = self::fTol();
        $fRtol = INF;
        $xTol = INF;
        $xRtol = INF;

        $func = static fn (array $z): array => array_map('floatval', array_values($F($z)));

        $x = array_map('floatval', array_values($xin));
        $n = count($x);
        $dx = array_fill(0, $n, INF);
        $Fx = $func($x);
        $FxNorm = self::n2($Fx);

        $jac = self::makeJacobian($jacobian);
        $jac->setup($x, $Fx, $func);

        if ($maxiter === null) {
            $maxiter = 100 * ($n + 1);
        }

        $gamma = 0.9;
        $etaMax = 0.9999;
        $etaThreshold = 0.1;
        $eta = 1e-3;
        $f0Norm = null;
        $converged = false;

        for ($it = 0; $it < $maxiter; $it++) {
            // TerminationCondition.check (max-norm)
            $fNorm = self::maxNorm($Fx);
            $xNorm = self::maxNorm($x);
            $dxNorm = self::maxNorm($dx);
            if ($f0Norm === null) {
                $f0Norm = $fNorm;
            }
            if ($fNorm === 0.0) {
                $converged = true;
                break;
            }
            $status = ($fNorm <= $fTol && $fNorm / $fRtol <= $f0Norm)
                && ($dxNorm <= $xTol && $dxNorm / $xRtol <= $xNorm);
            if ($status) {
                $converged = true;
                break;
            }

            $tol = min($eta, $eta * $FxNorm);
            $dx = self::scale($jac->solve($Fx, $tol), -1.0);
            if (self::n2($dx) === 0.0) {
                throw new \RuntimeException('Jacobian inversion yielded zero vector. This indicates a bug in the Jacobian approximation.');
            }

            [$s, $x, $Fx, $FxNormNew] = self::lineSearch($func, $x, $Fx, $dx);

            $jac->update($x, $Fx);

            $etaA = $gamma * $FxNormNew ** 2 / $FxNorm ** 2;
            if ($gamma * $eta ** 2 < $etaThreshold) {
                $eta = min($etaMax, $etaA);
            } else {
                $eta = min($etaMax, max($etaA, $gamma * $eta ** 2));
            }
            $FxNorm = $FxNormNew;
        }

        if (! $converged) {
            throw new NoConvergence($x);
        }

        return $x;
    }

    private static function makeJacobian(string $kind): JacobianApprox
    {
        return match ($kind) {
            'broyden1' => new BroydenFirst(),
            'broyden2' => new BroydenSecond(),
            'anderson' => new AndersonMixing(),
            'diagbroyden' => new DiagBroyden(),
            'linearmixing' => new LinearMixing(),
            'excitingmixing' => new ExcitingMixing(),
            'krylov' => new KrylovJacobian(),
            default => throw new \RuntimeException("unknown jacobian '{$kind}'"),
        };
    }

    /**
     * _nonlin_line_search with search_type='armijo'.
     *
     * @param callable(list<float>): list<float> $func
     * @param list<float> $x
     * @param list<float> $Fx
     * @param list<float> $dx
     * @return array{0: float, 1: list<float>, 2: list<float>, 3: float}
     */
    private static function lineSearch(callable $func, array $x, array $Fx, array $dx): array
    {
        $tmpS = 0.0;
        $tmpFx = $Fx;
        $tmpPhi = self::n2($Fx) ** 2;

        $phi = static function (float $s, bool $store = true) use (&$tmpS, &$tmpFx, &$tmpPhi, $func, $x, $dx): float {
            if ($s === $tmpS) {
                return $tmpPhi;
            }
            $xt = [];
            foreach ($x as $i => $xi) {
                $xt[] = $xi + $s * $dx[$i];
            }
            $v = $func($xt);
            $p = self::safeNorm($v) ** 2;
            if ($store) {
                $tmpS = $s;
                $tmpPhi = $p;
                $tmpFx = $v;
            }
            return $p;
        };

        $s = self::scalarSearchArmijo($phi, $tmpPhi, -$tmpPhi, 1e-4, 1.0, 1e-2);
        if ($s === null) {
            $s = 1.0;
        }
        $xNew = [];
        foreach ($x as $i => $xi) {
            $xNew[] = $xi + $s * $dx[$i];
        }
        $FxNew = ($s === $tmpS) ? $tmpFx : $func($xNew);

        return [$s, $xNew, $FxNew, self::n2($FxNew)];
    }

    /**
     * scipy.optimize._linesearch.scalar_search_armijo (returns the step, or null on failure).
     *
     * @param callable(float, bool=): float $phi
     */
    private static function scalarSearchArmijo(callable $phi, float $phi0, float $derphi0, float $c1, float $alpha0, float $amin): ?float
    {
        $phiA0 = $phi($alpha0);
        if ($phiA0 <= $phi0 + $c1 * $alpha0 * $derphi0) {
            return $alpha0;
        }
        $alpha1 = -$derphi0 * $alpha0 ** 2 / 2.0 / ($phiA0 - $phi0 - $derphi0 * $alpha0);
        $phiA1 = $phi($alpha1);
        if ($phiA1 <= $phi0 + $c1 * $alpha1 * $derphi0) {
            return $alpha1;
        }
        while ($alpha1 > $amin) {
            $factor = $alpha0 ** 2 * $alpha1 ** 2 * ($alpha1 - $alpha0);
            $a = $alpha0 ** 2 * ($phiA1 - $phi0 - $derphi0 * $alpha1)
                - $alpha1 ** 2 * ($phiA0 - $phi0 - $derphi0 * $alpha0);
            $a /= $factor;
            $b = -$alpha0 ** 3 * ($phiA1 - $phi0 - $derphi0 * $alpha1)
                + $alpha1 ** 3 * ($phiA0 - $phi0 - $derphi0 * $alpha0);
            $b /= $factor;
            $alpha2 = (-$b + sqrt(abs($b ** 2 - 3 * $a * $derphi0))) / (3.0 * $a);
            $phiA2 = $phi($alpha2);
            if ($phiA2 <= $phi0 + $c1 * $alpha2 * $derphi0) {
                return $alpha2;
            }
            if (($alpha1 - $alpha2) > $alpha1 / 2.0 || (1 - $alpha2 / $alpha1) < 0.96) {
                $alpha2 = $alpha1 / 2.0;
            }
            $alpha0 = $alpha1;
            $alpha1 = $alpha2;
            $phiA0 = $phiA1;
            $phiA1 = $phiA2;
        }
        return null;
    }

    /** 2-norm (scipy.linalg.norm default). @param list<float> $v */
    public static function n2(array $v): float
    {
        $s = 0.0;
        foreach ($v as $x) {
            $s += $x * $x;
        }
        return sqrt($s);
    }

    /** max-norm (scipy _nonlin.maxnorm). @param list<float> $v */
    private static function maxNorm(array $v): float
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

    /** _safe_norm: inf if any non-finite. @param list<float> $v */
    public static function safeNorm(array $v): float
    {
        foreach ($v as $x) {
            if (! is_finite($x)) {
                return INF;
            }
        }
        return self::n2($v);
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

    /** vdot. @param list<float> $a @param list<float> $b */
    public static function dot(array $a, array $b): float
    {
        $s = 0.0;
        foreach ($a as $i => $ai) {
            $s += $ai * $b[$i];
        }
        return $s;
    }

    /**
     * Gaussian elimination with partial pivoting (scipy.linalg.solve stand-in for
     * the small MxM systems inside the Jacobian approximations).
     *
     * @param list<list<float>> $A
     * @param list<float> $b
     * @return list<float>
     */
    public static function solveLin(array $A, array $b): array
    {
        $n = count($b);
        $M = [];
        for ($i = 0; $i < $n; $i++) {
            $M[$i] = array_values($A[$i]);
            $M[$i][$n] = $b[$i];
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
            if ($best === 0.0) {
                throw new \RuntimeException('singular matrix');
            }
            if ($piv !== $col) {
                $t = $M[$col];
                $M[$col] = $M[$piv];
                $M[$piv] = $t;
            }
            for ($r = 0; $r < $n; $r++) {
                if ($r !== $col) {
                    $f = $M[$r][$col] / $M[$col][$col];
                    for ($c = $col; $c <= $n; $c++) {
                        $M[$r][$c] -= $f * $M[$col][$c];
                    }
                }
            }
        }
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[$i] = $M[$i][$n] / $M[$i][$i];
        }
        return $out;
    }
}
