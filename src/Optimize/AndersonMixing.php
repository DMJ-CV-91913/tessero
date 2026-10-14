<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize._nonlin.Anderson (extended Anderson mixing). Maintains a rank-M
 * approximation of the inverse Jacobian over the last M iterate/residual deltas,
 * with Tikhonov regularization w0 = 0.01 (scipy defaults M = 5, w0 = 0.01).
 *
 * @internal
 */
final class AndersonMixing implements JacobianApprox
{
    private ?float $alpha;
    private int $M;
    private float $w0;
    /** @var list<list<float>> */
    private array $dxs = [];
    /** @var list<list<float>> */
    private array $dfs = [];
    /** @var list<list<float>> */
    private array $a = [];
    /** @var list<float> */
    private array $lastF = [];
    /** @var list<float> */
    private array $lastX = [];

    public function __construct(?float $alpha = null, float $w0 = 0.01, int $M = 5)
    {
        $this->alpha = $alpha;
        $this->w0 = $w0;
        $this->M = $M;
    }

    public function setup(array $x, array $F, callable $func): void
    {
        $this->lastF = $F;
        $this->lastX = $x;
        if ($this->alpha === null) {
            $normf0 = NonlinSolve::n2($F);
            $this->alpha = $normf0 ? 0.5 * max(NonlinSolve::n2($x), 1.0) / $normf0 : 1.0;
        }
    }

    public function solve(array $f, float $tol = 0.0): array
    {
        $alpha = (float) $this->alpha;
        $dx = [];
        foreach ($f as $fi) {
            $dx[] = -$alpha * $fi;
        }
        $n = count($this->dxs);
        if ($n === 0) {
            return $dx;
        }
        $dfF = [];
        for ($k = 0; $k < $n; $k++) {
            $dfF[] = NonlinSolve::dot($this->dfs[$k], $f);
        }
        try {
            $gamma = NonlinSolve::solveLin($this->a, $dfF);
        } catch (\RuntimeException) {
            // singular; reset the Jacobian approximation
            $this->dxs = [];
            $this->dfs = [];
            return $dx;
        }
        for ($m = 0; $m < $n; $m++) {
            $g = $gamma[$m];
            $dxm = $this->dxs[$m];
            $dfm = $this->dfs[$m];
            foreach ($dx as $i => $_) {
                $dx[$i] += $g * ($dxm[$i] + $alpha * $dfm[$i]);
            }
        }
        return $dx;
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

        if ($this->M !== 0) {
            $this->dxs[] = $dx;
            $this->dfs[] = $df;
            while (count($this->dxs) > $this->M) {
                array_shift($this->dxs);
                array_shift($this->dfs);
            }
            $n = count($this->dxs);
            $a = [];
            for ($i = 0; $i < $n; $i++) {
                $a[$i] = array_fill(0, $n, 0.0);
            }
            for ($i = 0; $i < $n; $i++) {
                for ($j = $i; $j < $n; $j++) {
                    $wd = ($i === $j) ? $this->w0 ** 2 : 0.0;
                    $a[$i][$j] = (1 + $wd) * NonlinSolve::dot($this->dfs[$i], $this->dfs[$j]);
                }
            }
            for ($i = 0; $i < $n; $i++) {
                for ($j = $i + 1; $j < $n; $j++) {
                    $a[$j][$i] = $a[$i][$j];
                }
            }
            $this->a = $a;
        }

        $this->lastF = $f;
        $this->lastX = $x;
    }
}
