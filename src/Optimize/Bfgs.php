<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize.BFGS Hessian update strategy (Nocedal & Wright p.140, formula 6.19
 * for the 'hess' path / 6.17 for 'inv_hess'). Pure-PHP port; matches scipy's BFGS
 * matrix sequence to machine precision.
 *
 * ctor mirrors scipy: ($exceptionStrategy='skip_update', $minCurvature=null,
 * $initScale='auto'). min_curvature defaults to 1e-8 (skip) or 0.2 (damp).
 */
final class Bfgs extends HessianUpdateStrategy
{
    private float $minCurvature;
    private string $exceptionStrategy;

    public function __construct(
        string $exceptionStrategy = 'skip_update',
        ?float $minCurvature = null,
        float|string $initScale = 'auto',
    ) {
        if ($exceptionStrategy === 'skip_update') {
            $this->minCurvature = $minCurvature ?? 1e-8;
        } elseif ($exceptionStrategy === 'damp_update') {
            $this->minCurvature = $minCurvature ?? 0.2;
        } else {
            throw new \InvalidArgumentException("`exception_strategy` must be 'skip_update' or 'damp_update'.");
        }
        $this->exceptionStrategy = $exceptionStrategy;
        parent::__construct($initScale);
    }

    protected function updateImplementation(array $dx, array $dg): void
    {
        // Hessian path: w = dx, z = dg; inverse path swaps them.
        if ($this->approxType === 'hess') {
            $w = $dx;
            $z = $dg;
        } else {
            $w = $dg;
            $z = $dx;
        }
        $wz = self::ddot($w, $z);
        $Mw = self::matVec($this->M, $w);
        $wMw = self::ddot($Mw, $w);
        if ($wMw <= 0.0) {
            // reinitialise with the auto scale, as scipy does on indefinite roundoff
            $scale = $this->autoScale($dx, $dg);
            $this->M = self::eye($this->n);
            for ($i = 0; $i < $this->n; $i++) {
                $this->M[$i][$i] = $scale;
            }
            $Mw = self::matVec($this->M, $w);
            $wMw = self::ddot($Mw, $w);
        }
        if ($wz <= $this->minCurvature * $wMw) {
            if ($this->exceptionStrategy === 'skip_update') {
                return;
            }
            // damp_update: interpolate between the BFGS result and the unmodified matrix
            $updateFactor = (1 - $this->minCurvature) / (1 - $wz / $wMw);
            $zNew = [];
            foreach ($z as $i => $zi) {
                $zNew[$i] = $updateFactor * $zi + (1 - $updateFactor) * $Mw[$i];
            }
            $z = $zNew;
            $wz = self::ddot($w, $z);
        }
        if ($this->approxType === 'hess') {
            // B <- B + z z^T / wz - Mw Mw^T / wMw
            $this->syr(1.0 / $wz, $z);
            $this->syr(-1.0 / $wMw, $Mw);
        } else {
            // H <- H - (1/ys)(s Hy^T + Hy s^T) + ((ys + yHy)/ys^2) s s^T
            $this->syr2(-1.0 / $wz, $z, $Mw);   // z = s, Mw = Hy here
            $this->syr(($wz + $wMw) / ($wz * $wz), $z);
        }
    }

    /**
     * Symmetric rank-1 update M <- M + alpha * v v^T (full symmetric, like BLAS syr + symmetrise).
     *
     * @param list<float> $v
     */
    private function syr(float $alpha, array $v): void
    {
        for ($i = 0; $i < $this->n; $i++) {
            for ($j = 0; $j < $this->n; $j++) {
                $this->M[$i][$j] += $alpha * $v[$i] * $v[$j];
            }
        }
    }

    /**
     * Symmetric rank-2 update M <- M + alpha * (a b^T + b a^T).
     *
     * @param list<float> $a
     * @param list<float> $b
     */
    private function syr2(float $alpha, array $a, array $b): void
    {
        for ($i = 0; $i < $this->n; $i++) {
            for ($j = 0; $j < $this->n; $j++) {
                $this->M[$i][$j] += $alpha * ($a[$i] * $b[$j] + $b[$i] * $a[$j]);
            }
        }
    }
}
