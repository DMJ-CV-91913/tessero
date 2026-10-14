<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * Pure-PHP port of scipy.optimize._hessian_update_strategy.FullHessianUpdateStrategy:
 * the dense quasi-Newton Hessian (or inverse-Hessian) approximation shared by BFGS
 * and SR1. Backend-neutral (plain PHP arrays, no NDArray/BLAS) so both Tessero
 * backends produce identical matrices.
 *
 * scipy stores only the upper triangle (BLAS syr/syr2 updates) and symmetrises on
 * get_matrix(); maintaining a full symmetric matrix here is numerically equivalent
 * (the rank-1/rank-2 updates are symmetric) and matches scipy to machine precision.
 *
 * The public surface mirrors scipy: initialize($n, $approxType), update($dx, $dg),
 * dot($p) (also via the `@`-equivalent), get_matrix().
 */
abstract class HessianUpdateStrategy
{
    /** @var list<list<float>>|null the stored n x n symmetric matrix (B for 'hess', H for 'inv_hess'). */
    protected ?array $M = null;
    protected int $n = 0;
    protected string $approxType = 'hess';
    protected bool $firstIteration = true;

    /** @var float|string 'auto' or an explicit scalar scale. */
    protected float|string $initScale;

    public function __construct(float|string $initScale = 'auto')
    {
        $this->initScale = $initScale;
    }

    /** Allocate the n x n identity for the Hessian ('hess') or inverse Hessian ('inv_hess'). */
    public function initialize(int $n, string $approxType = 'hess'): void
    {
        if ($approxType !== 'hess' && $approxType !== 'inv_hess') {
            throw new \InvalidArgumentException("`approx_type` must be 'hess' or 'inv_hess'.");
        }
        $this->n = $n;
        $this->approxType = $approxType;
        $this->firstIteration = true;
        $this->M = self::eye($n);
    }

    /**
     * Nocedal & Wright (6.20) auto-scale heuristic for the first iteration.
     *
     * @param list<float> $dx
     * @param list<float> $dg
     */
    protected function autoScale(array $dx, array $dg): float
    {
        $sNorm2 = self::ddot($dx, $dx);
        $yNorm2 = self::ddot($dg, $dg);
        $ys = abs(self::ddot($dg, $dx));
        if ($ys === 0.0 || $yNorm2 === 0.0 || $sNorm2 === 0.0) {
            return 1.0;
        }
        return $this->approxType === 'hess' ? $yNorm2 / $ys : $ys / $yNorm2;
    }

    /**
     * Update the internal matrix from a step dx = x2 - x1 and gradient change dg.
     *
     * @param list<float> $dx
     * @param list<float> $dg
     */
    public function update(array $dx, array $dg): void
    {
        $dx = array_map('floatval', array_values($dx));
        $dg = array_map('floatval', array_values($dg));
        if (self::allZero($dx)) {
            return;
        }
        if (self::allZero($dg)) {
            // scipy warns and skips (delta_grad == 0: function is locally linear).
            return;
        }
        if ($this->firstIteration) {
            $scale = $this->initScale === 'auto' ? $this->autoScale($dx, $dg) : (float) $this->initScale;
            for ($i = 0; $i < $this->n; $i++) {
                for ($j = 0; $j < $this->n; $j++) {
                    $this->M[$i][$j] *= $scale;
                }
            }
            $this->firstIteration = false;
        }
        $this->updateImplementation($dx, $dg);
    }

    /** @param list<float> $dx @param list<float> $dg */
    abstract protected function updateImplementation(array $dx, array $dg): void;

    /**
     * Product of the internal matrix with p (the `@`-operator in scipy).
     *
     * @param list<float> $p
     * @return list<float>
     */
    public function dot(array $p): array
    {
        $p = array_map('floatval', array_values($p));
        return self::matVec($this->M, $p);
    }

    /**
     * The current dense n x n matrix (symmetric).
     *
     * @return list<list<float>>
     */
    public function getMatrix(): array
    {
        return array_map(static fn (array $row): array => array_values($row), $this->M);
    }

    // --- small dense linear-algebra helpers (plain PHP) ---

    /** @return list<list<float>> */
    protected static function eye(int $n): array
    {
        $m = [];
        for ($i = 0; $i < $n; $i++) {
            $m[$i] = array_fill(0, $n, 0.0);
            $m[$i][$i] = 1.0;
        }
        return $m;
    }

    /** @param list<float> $a @param list<float> $b */
    protected static function ddot(array $a, array $b): float
    {
        $s = 0.0;
        foreach ($a as $i => $v) {
            $s += $v * $b[$i];
        }
        return $s;
    }

    /** @param list<float> $a */
    protected static function allZero(array $a): bool
    {
        foreach ($a as $v) {
            if ($v !== 0.0) {
                return false;
            }
        }
        return true;
    }

    /** @param list<list<float>> $M @param list<float> $v @return list<float> */
    protected static function matVec(array $M, array $v): array
    {
        $out = [];
        foreach ($M as $i => $row) {
            $out[$i] = self::ddot($row, $v);
        }
        return $out;
    }
}
