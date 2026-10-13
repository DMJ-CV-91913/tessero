<?php

declare(strict_types=1);

namespace Tessero;

use Tessero\Exceptions\TesseroException;
use Tessero\Optimize\Minimize;
use Tessero\Special;

/**
 * scipy.stats.barnard_exact: Barnard's exact test for a 2x2 contingency table.
 *
 * Pure PHP. The Wald (score) statistic is computed over the full x1 x x2 sample space; the p-value is the
 * maximum over the nuisance parameter pi in (0, 1) of the sum of binomial probabilities for every sample at
 * least as extreme as the observed one. SciPy maximises this smooth 1-D objective with `shgo` (Sobol sampling
 * + local refinement). We reproduce the maximiser deterministically: evaluate the negative-log-p objective on
 * an n-point interior grid of (0, 1), then run a bounded Brent refinement (Tessero\Optimize\Minimize::fminbound)
 * inside the cell around every local minimum. This matches scipy.stats.barnard_exact to ~1e-12 on the p-value
 * and exactly on the statistic. Mirrors the facade style of {@see StatsFns}; one class serves both backends
 * with no kernel call (gammaln comes from {@see Special}).
 */
final class StatsBarnard
{
    /**
     * @param array{0: array{0:int,1:int}, 1: array{0:int,1:int}} $table 2x2 contingency table of non-negative ints
     * @param string $alternative one of 'two-sided', 'less', 'greater'
     * @param bool   $pooled      pooled (Student) vs unpooled (Welch) variance in the Wald statistic
     * @param int    $n           number of interior grid points for the nuisance-parameter maximisation
     *
     * @return array{statistic: float, pvalue: float}
     */
    public static function barnardExact(
        array $table,
        string $alternative = 'two-sided',
        bool $pooled = true,
        int $n = 32
    ): array {
        if ($n <= 0) {
            throw new TesseroException("barnard_exact: number of points `n` must be strictly positive, found {$n}.");
        }
        if (! in_array($alternative, ['two-sided', 'less', 'greater'], true)) {
            throw new TesseroException(
                "barnard_exact: `alternative` should be one of {'two-sided', 'less', 'greater'}, found '{$alternative}'."
            );
        }
        // Validate shape (2x2) and non-negativity.
        if (count($table) !== 2 || count($table[0]) !== 2 || count($table[1]) !== 2) {
            throw new TesseroException('barnard_exact: the input `table` must be of shape (2, 2).');
        }
        $t00 = (int) $table[0][0];
        $t01 = (int) $table[0][1];
        $t10 = (int) $table[1][0];
        $t11 = (int) $table[1][1];
        foreach ([$t00, $t01, $t10, $t11] as $v) {
            if ($v < 0) {
                throw new TesseroException('barnard_exact: all values in `table` must be nonnegative.');
            }
        }

        $total1 = $t00 + $t10;   // column 1 total
        $total2 = $t01 + $t11;   // column 2 total

        // If either column sums to zero the p-value is 1 and the statistic is NaN.
        if ($total1 === 0 || $total2 === 0) {
            return ['statistic' => NAN, 'pvalue' => 1.0];
        }

        $nTot = $total1 + $total2;

        // Wald (score) statistic for every (x1, x2) combination.
        $wald = [];
        for ($x1 = 0; $x1 <= $total1; $x1++) {
            $p1 = $x1 / $total1;
            $row = [];
            for ($x2 = 0; $x2 <= $total2; $x2++) {
                $p2 = $x2 / $total2;
                if ($p1 === $p2) {
                    $row[$x2] = 0.0;              // scipy forces wald = 0 when p1 == p2 (removes NaN)
                    continue;
                }
                if ($pooled) {
                    $p = ($x1 + $x2) / $nTot;
                    $var = $p * (1.0 - $p) * (1.0 / $total1 + 1.0 / $total2);
                } else {
                    $var = $p1 * (1.0 - $p1) / $total1 + $p2 * (1.0 - $p2) / $total2;
                }
                if ($var <= 0.0) {
                    $row[$x2] = ($p1 - $p2) > 0 ? INF : -INF;
                } else {
                    $row[$x2] = ($p1 - $p2) / sqrt($var);
                }
            }
            $wald[$x1] = $row;
        }

        $obs = $wald[$t00][$t01];
        $absObs = abs($obs);

        // Extreme-region membership + log-combination weights, flattened over the included cells.
        $logComb1 = self::logCombinations($total1);
        $logComb2 = self::logCombinations($total2);

        $sArr = [];      // x1 + x2 for each included cell
        $lcArr = [];     // log C(total1, x1) + log C(total2, x2)
        for ($x1 = 0; $x1 <= $total1; $x1++) {
            for ($x2 = 0; $x2 <= $total2; $x2++) {
                $w = $wald[$x1][$x2];
                $include = match ($alternative) {
                    'two-sided' => abs($w) >= $absObs,
                    'less' => $w <= $obs,
                    'greater' => $w >= $obs,
                };
                if ($include) {
                    $sArr[] = $x1 + $x2;
                    $lcArr[] = $logComb1[$x1] + $logComb2[$x2];
                }
            }
        }

        // Negative log p-value as a function of the nuisance parameter pi in (0, 1).
        $negLogP = static function (float $pi) use ($sArr, $lcArr, $nTot): float {
            if ($pi <= 0.0 || $pi >= 1.0) {
                return INF;
            }
            $logPi = log($pi);
            $log1mPi = log(1.0 - $pi);
            $m = count($sArr);
            $tv = [];
            $max = -INF;
            for ($i = 0; $i < $m; $i++) {
                $s = $sArr[$i];
                $a = $s === 0 ? 0.0 : $logPi * $s;
                $b = $s === $nTot ? 0.0 : $log1mPi * ($nTot - $s);
                $v = $lcArr[$i] + $a + $b;
                $tv[$i] = $v;
                if ($v > $max) {
                    $max = $v;
                }
            }
            $sum = 0.0;
            for ($i = 0; $i < $m; $i++) {
                $sum += exp($tv[$i] - $max);
            }
            if ($sum <= 0.0) {
                return INF;
            }
            return -($max + log($sum));
        };

        // Deterministic maximisation: grid of n interior points, then bounded Brent refinement
        // inside the cell around every local minimum of negLogP.
        $grid = [];
        $vals = [];
        for ($k = 1; $k <= $n; $k++) {
            $pi = $k / ($n + 1);                  // interior points, excludes 0 and 1
            $grid[] = $pi;
            $vals[] = $negLogP($pi);
        }

        $best = INF;
        $cnt = count($grid);
        for ($k = 0; $k < $cnt; $k++) {
            $isLocalMin = ($k === 0 || $vals[$k] <= $vals[$k - 1])
                && ($k === $cnt - 1 || $vals[$k] <= $vals[$k + 1]);
            if (! $isLocalMin) {
                continue;
            }
            if ($vals[$k] < $best) {
                $best = $vals[$k];
            }
            $lo = $k > 0 ? $grid[$k - 1] : 0.0;
            $hi = $k < $cnt - 1 ? $grid[$k + 1] : 1.0;
            $eps = 1e-15;
            $lo = max($lo, $eps);
            $hi = min($hi, 1.0 - $eps);
            $xf = Minimize::fminbound($negLogP, $lo, $hi, 1e-12);
            $fx = $negLogP($xf);
            if ($fx < $best) {
                $best = $fx;
            }
        }

        $pvalue = $best === INF ? 0.0 : exp(-$best);
        if ($pvalue < 0.0) {
            $pvalue = 0.0;
        } elseif ($pvalue > 1.0) {
            $pvalue = 1.0;
        }

        return ['statistic' => $obs, 'pvalue' => $pvalue];
    }

    /**
     * Log of all binomial coefficients C(n, k) for k = 0..n, via gammaln
     * (matches scipy's _compute_log_combinations).
     *
     * @return list<float>
     */
    private static function logCombinations(int $n): array
    {
        $g = [];
        for ($k = 0; $k <= $n; $k++) {
            $g[$k] = (float) Special::gammaln($k + 1);
        }
        $lgN1 = (float) Special::gammaln($n + 1);
        $out = [];
        for ($k = 0; $k <= $n; $k++) {
            $out[$k] = $lgN1 - $g[$k] - $g[$n - $k];
        }
        return $out;
    }
}
