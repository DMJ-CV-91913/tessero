<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize._nonlin.LowRankMatrix: a matrix represented as
 * alpha*I + sum_n c_n d_n^T (real-valued). Once the rank reaches the vector
 * dimension it collapses to a dense matrix. Used by the Broyden Jacobians.
 *
 * @internal
 */
final class LowRankMatrix
{
    private ?float $alpha;
    private int $n;
    /** @var list<list<float>>|null */
    private ?array $cs = [];
    /** @var list<list<float>>|null */
    private ?array $ds = [];
    /** @var list<list<float>>|null */
    private ?array $collapsed = null;

    public function __construct(float $alpha, int $n)
    {
        $this->alpha = $alpha;
        $this->n = $n;
    }

    /**
     * @param list<float> $v
     * @param list<list<float>> $cs
     * @param list<list<float>> $ds
     * @return list<float>
     */
    private static function matvecLR(array $v, float $alpha, array $cs, array $ds): array
    {
        $w = [];
        foreach ($v as $x) {
            $w[] = $alpha * $x;
        }
        foreach ($cs as $k => $c) {
            $a = NonlinSolve::dot($ds[$k], $v);
            foreach ($v as $i => $_) {
                $w[$i] += $c[$i] * $a;
            }
        }
        return $w;
    }

    /**
     * (B + C D^H)^-1 via the matrix-inversion lemma.
     *
     * @param list<float> $v
     * @param list<list<float>> $cs
     * @param list<list<float>> $ds
     * @return list<float>
     */
    private static function solveLR(array $v, float $alpha, array $cs, array $ds): array
    {
        $m = count($cs);
        if ($m === 0) {
            $o = [];
            foreach ($v as $x) {
                $o[] = $x / $alpha;
            }
            return $o;
        }
        $A = [];
        for ($i = 0; $i < $m; $i++) {
            $row = array_fill(0, $m, 0.0);
            $row[$i] = $alpha;
            foreach ($cs as $j => $c) {
                $row[$j] += NonlinSolve::dot($ds[$i], $c);
            }
            $A[$i] = $row;
        }
        $q = [];
        foreach ($ds as $d) {
            $q[] = NonlinSolve::dot($d, $v) / $alpha;
        }
        $q = NonlinSolve::solveLin($A, $q);
        $w = [];
        foreach ($v as $x) {
            $w[] = $x / $alpha;
        }
        foreach ($cs as $k => $c) {
            $qc = $q[$k];
            foreach ($v as $i => $_) {
                $w[$i] -= $c[$i] * $qc;
            }
        }
        return $w;
    }

    /** @param list<float> $v @return list<float> */
    public function matvec(array $v): array
    {
        if ($this->collapsed !== null) {
            $out = [];
            foreach ($this->collapsed as $row) {
                $out[] = NonlinSolve::dot($row, $v);
            }
            return $out;
        }
        return self::matvecLR($v, (float) $this->alpha, $this->cs, $this->ds);
    }

    /** @param list<float> $v @return list<float> */
    public function rmatvec(array $v): array
    {
        if ($this->collapsed !== null) {
            $out = array_fill(0, $this->n, 0.0);
            for ($r = 0; $r < $this->n; $r++) {
                for ($i = 0; $i < $this->n; $i++) {
                    $out[$i] += $this->collapsed[$r][$i] * $v[$r];
                }
            }
            return $out;
        }
        return self::matvecLR($v, (float) $this->alpha, $this->ds, $this->cs);
    }

    /** @param list<float> $v @return list<float> */
    public function solve(array $v): array
    {
        if ($this->collapsed !== null) {
            return NonlinSolve::solveLin($this->collapsed, $v);
        }
        return self::solveLR($v, (float) $this->alpha, $this->cs, $this->ds);
    }

    /** @param list<float> $c @param list<float> $d */
    public function append(array $c, array $d): void
    {
        if ($this->collapsed !== null) {
            for ($i = 0; $i < $this->n; $i++) {
                for ($j = 0; $j < $this->n; $j++) {
                    $this->collapsed[$i][$j] += $c[$i] * $d[$j];
                }
            }
            return;
        }
        $this->cs[] = $c;
        $this->ds[] = $d;
        if (count($this->cs) > count($c)) {
            $this->collapse();
        }
    }

    private function collapse(): void
    {
        $alpha = (float) $this->alpha;
        $Gm = [];
        for ($i = 0; $i < $this->n; $i++) {
            $row = array_fill(0, $this->n, 0.0);
            $row[$i] = $alpha;
            $Gm[$i] = $row;
        }
        foreach ($this->cs as $k => $c) {
            $d = $this->ds[$k];
            for ($i = 0; $i < $this->n; $i++) {
                for ($j = 0; $j < $this->n; $j++) {
                    $Gm[$i][$j] += $c[$i] * $d[$j];
                }
            }
        }
        $this->collapsed = $Gm;
        $this->cs = null;
        $this->ds = null;
        $this->alpha = null;
    }

    /** 'restart' rank reduction (drops all columns once rank exceeds the limit). */
    public function restartReduce(float $rank): void
    {
        if ($this->collapsed !== null) {
            return;
        }
        if (count($this->cs) > $rank) {
            $this->cs = [];
            $this->ds = [];
        }
    }
}
