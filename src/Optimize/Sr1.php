<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize.SR1 symmetric-rank-1 Hessian update strategy (Nocedal & Wright
 * p.144-146). Pure-PHP port; matches scipy's SR1 matrix sequence to machine
 * precision. ctor mirrors scipy: ($minDenominator=1e-8, $initScale='auto').
 */
final class Sr1 extends HessianUpdateStrategy
{
    private float $minDenominator;

    public function __construct(float $minDenominator = 1e-8, float|string $initScale = 'auto')
    {
        $this->minDenominator = $minDenominator;
        parent::__construct($initScale);
    }

    protected function updateImplementation(array $dx, array $dg): void
    {
        if ($this->approxType === 'hess') {
            $w = $dx;
            $z = $dg;
        } else {
            $w = $dg;
            $z = $dx;
        }
        $Mw = self::matVec($this->M, $w);
        $zMinusMw = [];
        foreach ($z as $i => $zi) {
            $zMinusMw[$i] = $zi - $Mw[$i];
        }
        $denominator = self::ddot($w, $zMinusMw);
        $normW = sqrt(self::ddot($w, $w));
        $normR = sqrt(self::ddot($zMinusMw, $zMinusMw));
        if (abs($denominator) <= $this->minDenominator * $normW * $normR) {
            return;   // skip ill-conditioned update, as scipy does
        }
        // M <- M + (z - Mw)(z - Mw)^T / denominator
        $alpha = 1.0 / $denominator;
        for ($i = 0; $i < $this->n; $i++) {
            for ($j = 0; $j < $this->n; $j++) {
                $this->M[$i][$j] += $alpha * $zMinusMw[$i] * $zMinusMw[$j];
            }
        }
    }
}
