<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * Common interface for the quasi-Newton Jacobian approximations used by
 * {@see NonlinSolve} (scipy.optimize._nonlin.Jacobian): solve returns J^-1 v,
 * update refreshes the approximation at a new point.
 *
 * @internal
 */
interface JacobianApprox
{
    /**
     * @param list<float> $x
     * @param list<float> $F
     * @param callable(list<float>): list<float> $func
     */
    public function setup(array $x, array $F, callable $func): void;

    /** @param list<float> $f @return list<float> */
    public function solve(array $f, float $tol = 0.0): array;

    /** @param list<float> $x @param list<float> $f */
    public function update(array $x, array $f): void;
}
