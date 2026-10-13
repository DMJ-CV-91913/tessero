<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize.Bounds: box bounds on a minimisation variable. Scalars broadcast
 * to the problem dimension when used; stored here as given (lists or scalars) with
 * the same attribute surface as scipy (lb, ub, keep_feasible).
 */
final class Bounds
{
    /** @var list<float>|float */
    public readonly array|float $lb;
    /** @var list<float>|float */
    public readonly array|float $ub;
    /** @var list<bool>|bool */
    public readonly array|bool $keep_feasible;

    public function __construct(array|float|int $lb = -INF, array|float|int $ub = INF, array|bool $keep_feasible = false)
    {
        $this->lb = is_array($lb) ? array_map('floatval', array_values($lb)) : (float) $lb;
        $this->ub = is_array($ub) ? array_map('floatval', array_values($ub)) : (float) $ub;
        $this->keep_feasible = $keep_feasible;
    }

    /** Broadcast lb/ub to length n, matching how scipy applies Bounds to an n-vector. */
    public function resolved(int $n): array
    {
        $lb = is_array($this->lb) ? $this->lb : array_fill(0, $n, $this->lb);
        $ub = is_array($this->ub) ? $this->ub : array_fill(0, $n, $this->ub);
        return [$lb, $ub];
    }
}
