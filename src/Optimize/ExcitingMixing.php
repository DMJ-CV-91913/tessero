<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize._nonlin.ExcitingMixing: a tuned diagonal Jacobian approximation.
 * The inverse-Jacobian diagonal beta starts at alpha (autoscaled from the initial
 * residual as in scipy's GenericBroyden.setup when null) and is retuned each step:
 * entries whose residual kept its sign grow by alpha, the rest reset to alpha, all
 * clipped to [0, alphamax]. The step is -beta .* F.
 *
 * @internal
 */
final class ExcitingMixing implements JacobianApprox
{
    private ?float $alpha;
    private float $alphamax;
    /** @var list<float> diagonal of J^-1 */
    private array $beta = [];
    /** @var list<float> */
    private array $lastF = [];
    /** @var list<float> */
    private array $lastX = [];

    public function __construct(?float $alpha = null, float $alphamax = 1.0)
    {
        $this->alpha = $alpha;
        $this->alphamax = $alphamax;
    }

    public function setup(array $x, array $F, callable $func): void
    {
        $this->lastF = $F;
        $this->lastX = $x;
        if ($this->alpha === null) {
            $normf0 = NonlinSolve::n2($F);
            $this->alpha = $normf0 ? 0.5 * max(NonlinSolve::n2($x), 1.0) / $normf0 : 1.0;
        }
        $this->beta = array_fill(0, count($F), $this->alpha);
    }

    public function solve(array $f, float $tol = 0.0): array
    {
        $o = [];
        foreach ($f as $i => $fi) {
            $o[] = -$fi * $this->beta[$i];
        }
        return $o;
    }

    public function update(array $x, array $f): void
    {
        $a = $this->alpha;
        $amax = $this->alphamax;
        foreach ($f as $i => $fi) {
            // scipy: incr = f*last_f > 0; beta[incr]+=alpha; beta[~incr]=alpha; clip(0,alphamax)
            if ($fi * $this->lastF[$i] > 0.0) {
                $b = $this->beta[$i] + $a;
            } else {
                $b = $a;
            }
            if ($b < 0.0) {
                $b = 0.0;
            } elseif ($b > $amax) {
                $b = $amax;
            }
            $this->beta[$i] = $b;
        }
        $this->lastF = $f;
        $this->lastX = $x;
    }
}
