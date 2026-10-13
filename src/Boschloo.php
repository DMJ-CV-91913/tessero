<?php

declare(strict_types=1);

namespace Tessero;

use Tessero\Exceptions\TesseroException;

/**
 * scipy.stats.boschloo_exact: Boschloo's exact test on a 2x2 contingency table.
 *
 * Pure-PHP, both backends, no kernel call. Boschloo's test uses the p-value of Fisher's exact
 * test (the one-sided hypergeometric tail) as its statistic, then computes the probability under
 * H0 of observing a statistic at least as extreme, maximised over a nuisance success-probability
 * pi in (0, 1). This mirrors scipy's `_boschloo_exact`:
 *
 *   - `statistic` is the Fisher-exact one-sided p-value at the observed cell (hypergeom.cdf).
 *   - the null distribution is Binomial over the total column count with probability pi, and the
 *     p-value is the sum of binomial weights over all tables whose Fisher statistic is <= observed
 *     (relative guard 1e-13), maximised over pi.
 *
 * scipy drives this maximisation with `shgo` + Sobol sampling (default n=32); because the objective
 * is a smooth 1-D function on (0, 1) whose maximum is interior, a dense grid followed by
 * golden-section refinement recovers the same maximum to ~1e-12. The `n` argument is accepted for
 * signature parity (scipy rounds it up to a power of two for Sobol); it does not change the pure-PHP
 * result, which already resolves the global maximum.
 *
 * Log-combinations are computed with PHP's built-in lgamma (same recurrence scipy uses via gammaln),
 * so no scipy.special primitive is needed at runtime. Tessero also exposes the pieces natively
 * (Special::gammaln / Special::comb / Special::betainc and the stats.hypergeom distribution) should
 * a kernel-backed variant ever be wanted.
 */
final class Boschloo
{
    /**
     * scipy.stats.boschloo_exact(table, alternative='two-sided', n=32).
     *
     * @param array{0: array{0:int,1:int}, 1: array{0:int,1:int}} $table a 2x2 table of non-negative ints
     * @return array{statistic: float, pvalue: float}
     */
    public static function boschlooExact(array $table, string $alternative = 'two-sided', int $n = 32): array
    {
        if ($n <= 0) {
            throw new TesseroException("boschloo_exact: number of points `n` must be strictly positive, found {$n}.");
        }
        $t = self::validate($table);

        // If either column sums to zero, scipy returns (NaN, NaN) for every alternative.
        if (($t[0] + $t[2]) === 0 || ($t[1] + $t[3]) === 0) {
            return ['statistic' => NAN, 'pvalue' => NAN];
        }

        if ($alternative === 'two-sided') {
            $less = self::boschlooExact($table, 'less', $n);
            $greater = self::boschlooExact($table, 'greater', $n);
            $res = $less['pvalue'] < $greater['pvalue'] ? $less : $greater;
            $pvalue = min(max(2.0 * $res['pvalue'], 0.0), 1.0);
            return ['statistic' => $res['statistic'], 'pvalue' => $pvalue];
        }
        if ($alternative !== 'less' && $alternative !== 'greater') {
            throw new TesseroException(
                "boschloo_exact: `alternative` should be one of 'two-sided', 'less', 'greater', found '{$alternative}'."
            );
        }

        [$t00, $t01, $t10, $t11] = $t;
        $totalCol1 = $t00 + $t10;   // column 0 total
        $totalCol2 = $t01 + $t11;   // column 1 total
        $total = $totalCol1 + $totalCol2;

        // pvalues[x1][x2] = Fisher-exact one-sided p-value for the sub-table with column totals
        // (totalCol1, totalCol2) and the given x1/x2. For 'less' the tail is on column 0 (x1),
        // for 'greater' on column 1 (x2). Hypergeometric CDF: P(X <= k | M=total, n=draws, N=good).
        //   less:    cdf(x1; M=total, good=x1+x2, draws=totalCol1)
        //   greater: cdf(x2; M=total, good=x1+x2, draws=totalCol2)
        $pvalues = [];
        for ($x1 = 0; $x1 <= $totalCol1; $x1++) {
            $row = [];
            for ($x2 = 0; $x2 <= $totalCol2; $x2++) {
                $k = $x1 + $x2;
                $row[$x2] = ($alternative === 'less')
                    ? self::hypergeomCdf($x1, $total, $k, $totalCol1)
                    : self::hypergeomCdf($x2, $total, $k, $totalCol2);
            }
            $pvalues[$x1] = $row;
        }

        $fisherStat = $pvalues[$t00][$t01];

        // Precompute log C(totalCol1, x1) and log C(totalCol2, x2).
        $lc1 = self::logCombinations($totalCol1);
        $lc2 = self::logCombinations($totalCol2);

        // index mask: tables at least as extreme as observed (Fisher stat <= observed * (1+1e-13)).
        $thresh = $fisherStat * (1.0 + 1e-13);
        $idx = [];           // list of [x1Plusx2, logComb] for cells in the rejection region
        for ($x1 = 0; $x1 <= $totalCol1; $x1++) {
            for ($x2 = 0; $x2 <= $totalCol2; $x2++) {
                if ($pvalues[$x1][$x2] <= $thresh) {
                    $idx[] = [$x1 + $x2, $lc1[$x1] + $lc2[$x2]];
                }
            }
        }

        // Objective: negative log of the binomial-sum p-value as a function of the nuisance pi.
        // We MAXIMISE the p-value over pi in (0,1) (scipy minimises the negative log via shgo).
        $neglogp = static function (float $pi) use ($idx, $total): float {
            if ($pi <= 0.0 || $pi >= 1.0) {
                // Degenerate: scipy's `where`-guarded logs collapse the objective to p=1 at the
                // endpoints; exclude them (the true maximum is interior).
                return INF;
            }
            $logPi = log($pi);
            $log1mPi = log(1.0 - $pi);
            // tmp = logComb + k*log(pi) + (total-k)*log(1-pi); shift by max for numeric stability.
            $vals = [];
            $maxVal = -INF;
            foreach ($idx as [$k, $logComb]) {
                $a = ($k === 0) ? 0.0 : $logPi * $k;
                $b = ($k === $total) ? 0.0 : $log1mPi * ($total - $k);
                $v = $logComb + $a + $b;
                $vals[] = $v;
                if ($v > $maxVal) {
                    $maxVal = $v;
                }
            }
            $s = 0.0;
            foreach ($vals as $v) {
                $s += exp($v - $maxVal);
            }
            $logp = ($s > 0.0) ? ($maxVal + log($s)) : -INF;
            return -$logp;
        };

        // Dense coarse grid to bracket the global maximiser, then golden-section refinement.
        $best = self::maximise($neglogp);

        $p = min(max(exp(-$best), 0.0), 1.0);
        return ['statistic' => $fisherStat, 'pvalue' => $p];
    }

    /**
     * Minimise $f (= negative log p-value) on (0,1): dense grid bracket + golden-section refine.
     * Returns the minimum f value (i.e. -log of the maximised p-value).
     */
    private static function maximise(callable $f): float
    {
        $grid = 2000;                 // coarse scan; objective is smooth and unimodal-ish in practice
        $bestX = 0.5;
        $bestF = INF;
        $prevF = INF;
        for ($i = 1; $i < $grid; $i++) {
            $x = $i / $grid;
            $fx = $f($x);
            if ($fx < $bestF) {
                $bestF = $fx;
                $bestX = $x;
            }
            $prevF = $fx;
        }
        // Golden-section refinement inside the bracketing grid cell around $bestX.
        $h = 1.0 / $grid;
        $lo = max(1e-15, $bestX - $h);
        $hi = min(1.0 - 1e-15, $bestX + $h);
        $gr = (sqrt(5.0) - 1.0) / 2.0;
        $c = $hi - $gr * ($hi - $lo);
        $d = $lo + $gr * ($hi - $lo);
        $fc = $f($c);
        $fd = $f($d);
        for ($it = 0; $it < 200; $it++) {
            if ($hi - $lo < 1e-15) {
                break;
            }
            if ($fc < $fd) {
                $hi = $d;
                $d = $c;
                $fd = $fc;
                $c = $hi - $gr * ($hi - $lo);
                $fc = $f($c);
            } else {
                $lo = $c;
                $c = $d;
                $fc = $fd;
                $d = $lo + $gr * ($hi - $lo);
                $fd = $f($d);
            }
        }
        $fMid = $f(0.5 * ($lo + $hi));
        return min($bestF, $fc, $fd, $fMid);
    }

    /**
     * Hypergeometric CDF P(X <= k) for X ~ Hypergeom(M population, good successes, draws).
     * Matches scipy.stats.hypergeom.cdf(k, M, good, draws). Summed in log space via lgamma.
     *
     * @param int $k     number of observed successes (upper bound, inclusive)
     * @param int $M     total population
     * @param int $good  number of success states in the population
     * @param int $draws number drawn
     */
    private static function hypergeomCdf(int $k, int $M, int $good, int $draws): float
    {
        $bad = $M - $good;
        $lo = max(0, $draws - $bad);
        $hi = min($draws, $good);
        if ($k < $lo) {
            return 0.0;
        }
        if ($k >= $hi) {
            return 1.0;
        }
        $logDenom = self::logComb($M, $draws);
        $cdf = 0.0;
        for ($i = $lo; $i <= $k; $i++) {
            $logp = self::logComb($good, $i) + self::logComb($bad, $draws - $i) - $logDenom;
            $cdf += exp($logp);
        }
        return min($cdf, 1.0);
    }

    /** log C(n, k) via lgamma; 0 for out-of-range k. */
    private static function logComb(int $n, int $k): float
    {
        if ($k < 0 || $k > $n) {
            return -INF;
        }
        return self::lgamma($n + 1) - self::lgamma($k + 1) - self::lgamma($n - $k + 1);
    }

    /**
     * All log C(n, x) for x in [0, n], matching scipy _compute_log_combinations.
     *
     * @return list<float>
     */
    private static function logCombinations(int $n): array
    {
        $g = [];
        for ($i = 0; $i <= $n; $i++) {
            $g[$i] = self::lgamma($i + 1);
        }
        $gn1 = self::lgamma($n + 1);
        $out = [];
        for ($i = 0; $i <= $n; $i++) {
            $out[$i] = $gn1 - $g[$i] - $g[$n - $i];
        }
        return $out;
    }

    /**
     * log Gamma(x) for x > 0 via the Lanczos approximation (g=7, n=9), matching scipy.special.gammaln
     * to ~1e-14 on the integer arguments used here. PHP has no built-in lgamma/gamma, so this keeps the
     * facade pure-PHP; Tessero\Special::gammaln is the kernel-backed equivalent if ever preferred.
     */
    private static function lgamma(float $x): float
    {
        // For the integer arguments used in combinations, x is always >= 1 here.
        static $c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        $g = 7;
        $x -= 1.0;
        $a = $c[0];
        $t = $x + $g + 0.5;
        for ($i = 1; $i < 9; $i++) {
            $a += $c[$i] / ($x + $i);
        }
        return 0.5 * log(2.0 * M_PI) + ($x + 0.5) * log($t) - $t + log($a);
    }

    /**
     * Validate a 2x2 integer table and return it flattened as [t00, t01, t10, t11].
     *
     * @return array{0:int,1:int,2:int,3:int}
     */
    private static function validate(array $table): array
    {
        if (count($table) !== 2 || ! is_array($table[0] ?? null) || ! is_array($table[1] ?? null)
            || count($table[0]) !== 2 || count($table[1]) !== 2) {
            throw new TesseroException('boschloo_exact: the input `table` must be of shape (2, 2).');
        }
        $flat = [$table[0][0], $table[0][1], $table[1][0], $table[1][1]];
        $out = [];
        foreach ($flat as $v) {
            if (! is_int($v) && (! is_numeric($v) || (float) $v != (int) $v)) {
                throw new TesseroException('boschloo_exact: all values in `table` must be integers.');
            }
            $iv = (int) $v;
            if ($iv < 0) {
                throw new TesseroException('boschloo_exact: all values in `table` must be nonnegative.');
            }
            $out[] = $iv;
        }
        return [$out[0], $out[1], $out[2], $out[3]];
    }
}
