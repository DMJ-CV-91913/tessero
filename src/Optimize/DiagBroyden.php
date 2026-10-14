<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize._nonlin.DiagBroyden: diagonal Broyden Jacobian approximation
 * (retains only the diagonal of the Broyden matrix).
 *
 * @internal
 */
final class DiagBroyden implements JacobianApprox
{
    private ?float $alpha;
    /** @var list<float> diagonal of J */
    private array $d = [];
    /** @var list<float> */
    private array $lastF = [];
    /** @var list<float> */
    private array $lastX = [];

    public function __construct(?float $alpha = null)
    {
        $this->alpha = $alpha;
    }

    public function setup(array $x, array $F, callable $func): void
    {
        $this->lastF = $F;
        $this->lastX = $x;
        if ($this->alpha === null) {
            $normf0 = NonlinSolve::n2($F);
            $this->alpha = $normf0 ? 0.5 * max(NonlinSolve::n2($x), 1.0) / $normf0 : 1.0;
        }
        $this->d = array_fill(0, count($F), 1.0 / $this->alpha);
    }

    public function solve(array $f, float $tol = 0.0): array
    {
        $o = [];
        foreach ($f as $i => $fi) {
            $o[] = -$fi / $this->d[$i];
        }
        return $o;
    }

    public function update(array $x, array $f): void
    {
        $df = [];
        $dx = [];
        foreach ($f as $i => $fi) {
            $df[] = $fi - $this->lastF[$i];
        }
        foreach ($x as $i => $xi) {
            $dx[] = $xi - $this->lastX[$i];
        }
        $dxNorm = NonlinSolve::n2($dx);
        $dn2 = $dxNorm ** 2;
        foreach ($this->d as $i => $di) {
            $this->d[$i] = $di - ($df[$i] + $di * $dx[$i]) * $dx[$i] / $dn2;
        }
        $this->lastF = $f;
        $this->lastX = $x;
    }
}
