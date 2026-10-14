<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize._nonlin.LinearMixing: a scalar Jacobian approximation whose
 * inverse is the constant -alpha (so each step is -alpha * F). alpha is
 * autoscaled from the initial residual exactly as scipy's GenericBroyden.setup
 * when left null. The approximation is never updated (scipy's _update is a no-op).
 *
 * @internal
 */
final class LinearMixing implements JacobianApprox
{
    private ?float $alpha;

    public function __construct(?float $alpha = null)
    {
        $this->alpha = $alpha;
    }

    public function setup(array $x, array $F, callable $func): void
    {
        if ($this->alpha === null) {
            $normf0 = NonlinSolve::n2($F);
            $this->alpha = $normf0 ? 0.5 * max(NonlinSolve::n2($x), 1.0) / $normf0 : 1.0;
        }
    }

    public function solve(array $f, float $tol = 0.0): array
    {
        $a = $this->alpha;
        $o = [];
        foreach ($f as $fi) {
            $o[] = -$fi * $a;
        }
        return $o;
    }

    public function update(array $x, array $f): void
    {
        // scipy LinearMixing._update is a no-op (fixed scalar Jacobian).
    }
}
