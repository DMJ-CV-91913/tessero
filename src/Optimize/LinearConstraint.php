<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize.LinearConstraint: lb <= A @ x <= ub. A thin value class holding the
 * constraint matrix and its (broadcast-on-use) bounds, matching scipy's attributes
 * (A, lb, ub, keep_feasible). Defaults: lb = -inf, ub = +inf.
 */
final class LinearConstraint
{
    /** @var list<list<float>> */
    public readonly array $A;
    /** @var list<float>|float */
    public readonly array|float $lb;
    /** @var list<float>|float */
    public readonly array|float $ub;
    /** @var list<bool>|bool */
    public readonly array|bool $keep_feasible;

    public function __construct(array $A, array|float|int $lb = -INF, array|float|int $ub = INF, array|bool $keep_feasible = false)
    {
        // normalise A to a list of float rows (accept a 1-D row too)
        $rows = (isset($A[0]) && is_array($A[0])) ? $A : [$A];
        $this->A = array_map(static fn ($r): array => array_map('floatval', array_values($r)), array_values($rows));
        $this->lb = is_array($lb) ? array_map('floatval', array_values($lb)) : (float) $lb;
        $this->ub = is_array($ub) ? array_map('floatval', array_values($ub)) : (float) $ub;
        $this->keep_feasible = $keep_feasible;
    }
}
