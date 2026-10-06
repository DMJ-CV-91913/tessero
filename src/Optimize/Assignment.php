<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize.linear_sum_assignment: the optimal assignment of rows to columns of a cost matrix, by the
 * Hungarian algorithm (the O(n^2 m) shortest-augmenting-path form with dual potentials). The optimum is unique
 * for a cost matrix with no ties, so the result matches scipy.
 */
final class Assignment
{
    /**
     * @param list<list<float>> $cost  the cost matrix (nr x nc)
     * @return array{0: list<int>, 1: list<int>}  [row_ind, col_ind], row_ind ascending
     */
    public static function linearSumAssignment(array $cost, bool $maximize = false): array
    {
        $nr = count($cost);
        $nc = $nr > 0 ? count($cost[0]) : 0;
        if ($nr === 0 || $nc === 0) {
            return [[], []];
        }
        // work with (rows <= cols); transpose otherwise, and undo the mapping at the end
        $transposed = false;
        if ($nr > $nc) {
            $t = [];
            for ($j = 0; $j < $nc; $j++) {
                $t[$j] = [];
                for ($i = 0; $i < $nr; $i++) {
                    $t[$j][$i] = $cost[$i][$j];
                }
            }
            $cost = $t;
            [$nr, $nc] = [$nc, $nr];
            $transposed = true;
        }
        if ($maximize) {
            for ($i = 0; $i < $nr; $i++) {
                for ($j = 0; $j < $nc; $j++) {
                    $cost[$i][$j] = -$cost[$i][$j];
                }
            }
        }
        $n = $nr;
        $m = $nc;
        $u = array_fill(0, $n + 1, 0.0);
        $v = array_fill(0, $m + 1, 0.0);
        $p = array_fill(0, $m + 1, 0);        // p[j] = row (1-based) matched to column j
        $way = array_fill(0, $m + 1, 0);
        for ($i = 1; $i <= $n; $i++) {
            $p[0] = $i;
            $j0 = 0;
            $minv = array_fill(0, $m + 1, INF);
            $used = array_fill(0, $m + 1, false);
            do {
                $used[$j0] = true;
                $i0 = $p[$j0];
                $delta = INF;
                $j1 = -1;
                for ($j = 1; $j <= $m; $j++) {
                    if (!$used[$j]) {
                        $cur = $cost[$i0 - 1][$j - 1] - $u[$i0] - $v[$j];
                        if ($cur < $minv[$j]) {
                            $minv[$j] = $cur;
                            $way[$j] = $j0;
                        }
                        if ($minv[$j] < $delta) {
                            $delta = $minv[$j];
                            $j1 = $j;
                        }
                    }
                }
                for ($j = 0; $j <= $m; $j++) {
                    if ($used[$j]) {
                        $u[$p[$j]] += $delta;
                        $v[$j] -= $delta;
                    } else {
                        $minv[$j] -= $delta;
                    }
                }
                $j0 = $j1;
            } while ($p[$j0] !== 0);
            do {
                $j1 = $way[$j0];
                $p[$j0] = $p[$j1];
                $j0 = $j1;
            } while ($j0 !== 0);
        }
        // p[j] = row (1-based) assigned to column j; build the row->col map
        $colForRow = array_fill(0, $n, -1);
        for ($j = 1; $j <= $m; $j++) {
            if ($p[$j] !== 0) {
                $colForRow[$p[$j] - 1] = $j - 1;
            }
        }
        if (!$transposed) {
            $rowInd = range(0, $n - 1);
            $colInd = [];
            for ($i = 0; $i < $n; $i++) {
                $colInd[] = $colForRow[$i];
            }

            return [$rowInd, $colInd];
        }
        // undo the transpose: colForRow maps (original column -> original row); re-pair and sort by original row
        $pairs = [];
        for ($i = 0; $i < $n; $i++) {
            $pairs[] = [$colForRow[$i], $i];   // [original row, original column]
        }
        sort($pairs);
        $rowInd = [];
        $colInd = [];
        foreach ($pairs as $pr) {
            $rowInd[] = $pr[0];
            $colInd[] = $pr[1];
        }

        return [$rowInd, $colInd];
    }
}
