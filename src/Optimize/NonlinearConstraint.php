<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize.NonlinearConstraint: lb <= fun(x) <= ub for a vector-valued constraint
 * function. A value class carrying the callable and its bounds plus the Jacobian/Hessian
 * strategy slots scipy exposes (jac, hess, keep_feasible). Defaults mirror scipy.
 */
final class NonlinearConstraint
{
    /** @param callable(list<float>): (list<float>|float) $fun */
    public function __construct(
        public readonly mixed $fun,
        public readonly array|float|int $lb,
        public readonly array|float|int $ub,
        public readonly mixed $jac = '2-point',
        public readonly mixed $hess = null,
        public readonly array|bool $keep_feasible = false,
        public readonly float $finite_diff_rel_step = NAN,
    ) {
    }
}
