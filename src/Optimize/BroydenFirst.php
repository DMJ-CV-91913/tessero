<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize._nonlin.BroydenFirst ("good" Broyden): inverse-Jacobian
 * quasi-Newton update stored as a {@see LowRankMatrix}. max_rank = +inf with
 * 'restart' reduction, so the rank grows (and collapses to dense at dim).
 *
 * @internal
 */
class BroydenFirst implements JacobianApprox
{
    protected ?float $alpha;
    protected LowRankMatrix $Gm;
    protected float $maxRank = INF;
    /** @var list<float> */
    protected array $lastF = [];
    /** @var list<float> */
    protected array $lastX = [];
    /** @var callable(list<float>): list<float> */
    protected $func;

    public function __construct(?float $alpha = null)
    {
        $this->alpha = $alpha;
    }

    public function setup(array $x, array $F, callable $func): void
    {
        $this->func = $func;
        $this->lastF = $F;
        $this->lastX = $x;
        if ($this->alpha === null) {
            $normf0 = NonlinSolve::n2($F);
            $this->alpha = $normf0 ? 0.5 * max(NonlinSolve::n2($x), 1.0) / $normf0 : 1.0;
        }
        $this->Gm = new LowRankMatrix(-$this->alpha, count($F));
    }

    public function solve(array $f, float $tol = 0.0): array
    {
        $r = $this->Gm->matvec($f);
        foreach ($r as $v) {
            if (! is_finite($v)) {
                // singular; reset the Jacobian approximation
                $this->setup($this->lastX, $this->lastF, $this->func);
                return $this->Gm->matvec($f);
            }
        }
        return $r;
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
        $this->doUpdate($x, $f, $dx, $df, NonlinSolve::n2($dx), NonlinSolve::n2($df));
        $this->lastF = $f;
        $this->lastX = $x;
    }

    /**
     * @param list<float> $x
     * @param list<float> $f
     * @param list<float> $dx
     * @param list<float> $df
     */
    protected function doUpdate(array $x, array $f, array $dx, array $df, float $dxNorm, float $dfNorm): void
    {
        $this->Gm->restartReduce($this->maxRank - 1);
        $v = $this->Gm->rmatvec($dx);
        $gmdf = $this->Gm->matvec($df);
        $c = [];
        foreach ($dx as $i => $dxi) {
            $c[] = $dxi - $gmdf[$i];
        }
        $denom = NonlinSolve::dot($df, $v);
        $d = [];
        foreach ($v as $vi) {
            $d[] = $vi / $denom;
        }
        $this->Gm->append($c, $d);
    }
}
