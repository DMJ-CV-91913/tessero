<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Random\MT19937;
use Tessero\Special;

/**
 * scipy.stats.multiscale_graphcorr — the Multiscale Graph Correlation (MGC) independence test.
 *
 * Reproduces scipy's `_mgc.py` (statistic) and `scipy/stats/_stats.pyx::_local_correlations`
 * (the MGC-map), together with the permutation p-value. The statistic is deterministic; the
 * p-value draws, for each of `reps` replications, four uint32 seeds from a top-level MT19937 and
 * uses them to seed a child {@see MT19937} RandomState whose `permutation(n)` reorders `y`.
 *
 * The MT19937 stream and the MGC statistic have both been verified bit-for-bit against scipy
 * (`dstat = 0`, identical integer p-value numerator) over int seeds, large seeds and the
 * `RandomState(seed)` form.
 *
 * Pure-PHP on both backends: the distance centering / ranking / covariance accumulation and the
 * connected-component threshold are plain float64 array math; only `beta.ppf` routes through the
 * {@see Special} facade (and is identical on both backends through the regularised incomplete beta).
 */
final class MultiscaleGraphcorr
{
    /**
     * @param list<list<float>> $x (n, p) data or (n, n) distance matrix
     * @param list<list<float>> $y (n, q) data or (n, n) distance matrix
     * @param bool $computeDistance when true, euclidean distance matrices are computed from x, y;
     *                              when false, x and y are already distance matrices
     * @param int|MT19937 $randomState an int seed (→ RandomState(seed)) or an MT19937 instance
     *
     * @return array{statistic: float, pvalue: float, mgc_map: list<list<float>>, opt_scale: list<int>, null_dist: list<float>}
     */
    public static function multiscaleGraphcorr(
        array $x,
        array $y,
        bool $computeDistance = true,
        int $reps = 1000,
        int|MT19937 $randomState = 0,
        bool $isTwosamp = false,
    ): array {
        $x = self::as2d($x);
        $y = self::as2d($y);

        $nx = count($x);
        $ny = count($y);
        $px = count($x[0]);
        $py = count($y[0]);

        if ($nx !== $ny) {
            if ($px === $py) {
                $isTwosamp = true;
            } else {
                throw new \InvalidArgumentException('Shape mismatch: x and y must be [n,p]/[n,q] or [n,p]/[m,p].');
            }
        }
        if ($nx < 5 || $ny < 5) {
            throw new \InvalidArgumentException('MGC requires at least 5 samples.');
        }

        if ($isTwosamp) {
            if (! $computeDistance) {
                throw new \InvalidArgumentException('Cannot run two-sample if inputs are distance matrices.');
            }
            [$x, $y] = self::twoSampleTransform($x, $y);
        }

        if ($computeDistance) {
            $distx = self::euclideanDist($x);
            $disty = self::euclideanDist($y);
        } else {
            $distx = $x;
            $disty = $y;
        }

        [$stat, $statMap, $optScale] = self::mgcStat($distx, $disty);

        // ---- permutation p-value ----
        $outer = $randomState instanceof MT19937 ? $randomState : new MT19937($randomState);
        $n = count($disty);
        $null = [];
        for ($r = 0; $r < $reps; $r++) {
            $seed = $outer->randint(0, 1 << 32, 4); // randint(0, 2**32, size=4, dtype=uint32)
            $child = new MT19937($seed);
            $order = $child->permutation($n);
            $permy = self::permuteSym($disty, $order);
            [$ps] = self::mgcStat($distx, $permy);
            $null[] = $ps;
        }
        $ge = 0;
        foreach ($null as $v) {
            if ($v >= $stat) {
                $ge++;
            }
        }
        $pvalue = (1 + $ge) / (1 + $reps);

        return [
            'statistic' => $stat,
            'pvalue' => $pvalue,
            'mgc_map' => $statMap,
            'opt_scale' => $optScale,
            'null_dist' => $null,
        ];
    }

    /**
     * @param list<list<float>> $distx
     * @param list<list<float>> $disty
     *
     * @return array{0: float, 1: list<list<float>>, 2: list<int>}
     */
    private static function mgcStat(array $distx, array $disty): array
    {
        $smap = self::localCorrelations($distx, $disty);
        $m = count($smap);
        $nn = count($smap[0]);

        if ($m === 1 || $nn === 1) {
            $stat = $smap[$m - 1][$nn - 1];

            return [$stat, $smap, [$m, $nn]];
        }

        $samp = count($distx) - 1;
        $perSig = 1 - (0.02 / $samp);
        $thrBeta = $samp * ($samp - 3) / 4 - 0.5;
        // scipy: beta.ppf(perSig, thrBeta, thrBeta) * 2 - 1 ; beta.ppf(q,a,b) = betaincinv(a,b,q)
        $threshold = (float) Special::betaincinv($thrBeta, $thrBeta, $perSig) * 2 - 1;
        $threshold = max($threshold, $smap[$m - 1][$nn - 1]);

        // boolean significance mask
        $sig = [];
        $anySig = false;
        for ($i = 0; $i < $m; $i++) {
            $row = [];
            for ($j = 0; $j < $nn; $j++) {
                $b = $smap[$i][$j] > $threshold;
                $row[] = $b;
                if ($b) {
                    $anySig = true;
                }
            }
            $sig[] = $row;
        }

        if ($anySig) {
            $sig = self::largestComponent($sig, $m, $nn);
        } else {
            $sig = [[false]];
        }

        return self::smoothMgcMap($sig, $smap, $m, $nn);
    }

    /**
     * Largest 4-connected component of a boolean mask (scipy.ndimage.label default structure),
     * returned as a boolean mask selecting that component.
     *
     * @param list<list<bool>> $sig
     *
     * @return list<list<bool>>
     */
    private static function largestComponent(array $sig, int $m, int $nn): array
    {
        $lab = array_fill(0, $m, array_fill(0, $nn, 0));
        $counts = [];
        $next = 0;
        for ($i = 0; $i < $m; $i++) {
            for ($j = 0; $j < $nn; $j++) {
                if (! $sig[$i][$j] || $lab[$i][$j] !== 0) {
                    continue;
                }
                $next++;
                $size = 0;
                $stack = [[$i, $j]];
                $lab[$i][$j] = $next;
                while ($stack) {
                    [$ci, $cj] = array_pop($stack);
                    $size++;
                    foreach ([[-1, 0], [1, 0], [0, -1], [0, 1]] as [$di, $dj]) {
                        $ni = $ci + $di;
                        $nj = $cj + $dj;
                        if ($ni >= 0 && $ni < $m && $nj >= 0 && $nj < $nn
                            && $sig[$ni][$nj] && $lab[$ni][$nj] === 0) {
                            $lab[$ni][$nj] = $next;
                            $stack[] = [$ni, $nj];
                        }
                    }
                }
                $counts[$next] = $size;
            }
        }
        // scipy: max_label = argmax(label_counts[1:]) + 1 → first label with the maximum count
        $maxLabel = 1;
        $maxCount = -1;
        for ($l = 1; $l <= $next; $l++) {
            if (($counts[$l] ?? 0) > $maxCount) {
                $maxCount = $counts[$l];
                $maxLabel = $l;
            }
        }
        $out = [];
        for ($i = 0; $i < $m; $i++) {
            $row = [];
            for ($j = 0; $j < $nn; $j++) {
                $row[] = ($lab[$i][$j] === $maxLabel);
            }
            $out[] = $row;
        }

        return $out;
    }

    /**
     * @param list<list<bool>> $sig
     * @param list<list<float>> $smap
     *
     * @return array{0: float, 1: list<list<float>>, 2: list<int>}
     */
    private static function smoothMgcMap(array $sig, array $smap, int $m, int $nn): array
    {
        $stat = $smap[$m - 1][$nn - 1];
        $optScale = [$m, $nn];

        // np.linalg.norm(sig) != 0  <=>  any true entry
        $sigSum = 0;
        $sm = count($sig);
        $snn = count($sig[0]);
        for ($i = 0; $i < $sm; $i++) {
            for ($j = 0; $j < $snn; $j++) {
                if ($sig[$i][$j]) {
                    $sigSum++;
                }
            }
        }
        if ($sigSum !== 0) {
            if ($sigSum >= ceil(0.02 * max($m, $nn)) * min($m, $nn)) {
                // max over the significant region
                $maxCorr = -INF;
                for ($i = 0; $i < $m; $i++) {
                    for ($j = 0; $j < $nn; $j++) {
                        if ($sig[$i][$j] && $smap[$i][$j] > $maxCorr) {
                            $maxCorr = $smap[$i][$j];
                        }
                    }
                }
                if ($maxCorr >= $stat) {
                    $stat = $maxCorr;
                    // 1-D indices k*n+l of all (>= maxCorr) & sig, take the max
                    $maxOneD = -1;
                    for ($i = 0; $i < $m; $i++) {
                        for ($j = 0; $j < $nn; $j++) {
                            if ($smap[$i][$j] >= $maxCorr && $sig[$i][$j]) {
                                $idx = $i * $nn + $j;
                                if ($idx > $maxOneD) {
                                    $maxOneD = $idx;
                                }
                            }
                        }
                    }
                    $k = intdiv($maxOneD, $nn);
                    $l = $maxOneD % $nn;
                    $optScale = [$k + 1, $l + 1];
                }
            }
        }

        return [$stat, $smap, $optScale];
    }

    /**
     * Exact port of scipy/stats/_stats.pyx::_local_correlations(global_corr='mgc').
     *
     * @param list<list<float>> $distx
     * @param list<list<float>> $disty
     *
     * @return list<list<float>>
     */
    private static function localCorrelations(array $distx, array $disty): array
    {
        [$cdx, $rdx] = self::centerDistanceMatrix($distx);
        [$cdy, $rdy] = self::centerDistanceMatrix($disty);

        $cov = self::localCovariance($cdx, self::transpose($cdy), $rdx, self::transpose($rdy));
        $vxMat = self::localCovariance($cdx, self::transpose($cdx), $rdx, self::transpose($rdx));
        $vyMat = self::localCovariance($cdy, self::transpose($cdy), $rdy, self::transpose($rdy));
        $vx = self::diagonal($vxMat);
        $vy = self::diagonal($vyMat);

        $nx = count($cov);
        $ny = count($cov[0]);
        $corr = [];
        for ($k = 0; $k < $nx; $k++) {
            $row = [];
            for ($l = 0; $l < $ny; $l++) {
                $den = sqrt($vx[$k] * $vy[$l]);
                $c = $den > 0.0 ? $cov[$k][$l] / $den : NAN;
                if (! is_nan($c) && $c > 1.0) {
                    $c = 1.0;
                }
                $row[] = $c;
            }
            $corr[] = $row;
        }
        // corr_mat[local_varx <= 0, :] = 0 ; corr_mat[:, local_vary <= 0] = 0
        for ($k = 0; $k < $nx; $k++) {
            if ($vx[$k] <= 0.0) {
                for ($l = 0; $l < $ny; $l++) {
                    $corr[$k][$l] = 0.0;
                }
            }
        }
        for ($l = 0; $l < $ny; $l++) {
            if ($vy[$l] <= 0.0) {
                for ($k = 0; $k < $nx; $k++) {
                    $corr[$k][$l] = 0.0;
                }
            }
        }
        // any remaining NaN (den==0 but var>0 cannot happen; guard anyway) -> leave as produced

        return $corr;
    }

    /**
     * @param list<list<float>> $distx
     *
     * @return array{0: list<list<float>>, 1: list<list<int>>} centered matrix and dense ranks
     */
    private static function centerDistanceMatrix(array $distx): array
    {
        $n = count($distx);
        $rank = self::rankDistanceMatrix($distx);

        // exp = repeat(colmean * n/(n-1), n).reshape(-1,n).T
        //   => exp[i][j] = colmean[i] * n/(n-1)  (constant across j within a transposed row)
        // In scipy: exp_distx[i][j] = (mean over axis0 of column i) * n/(n-1).
        $colMean = array_fill(0, $n, 0.0);
        for ($j = 0; $j < $n; $j++) {
            $s = 0.0;
            for ($i = 0; $i < $n; $i++) {
                $s += $distx[$i][$j];
            }
            $colMean[$j] = $s / $n;
        }
        $scale = $n / ($n - 1);
        $cent = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $n; $j++) {
                // exp_distx = repeat(mean(axis=0)*n/(n-1), n).reshape(-1,n).T
                //   repeat->reshape(-1,n) gives R[r][c]=colMean[r]*scale; .T => exp[i][j]=colMean[j]*scale
                $row[] = $distx[$i][$j] - $colMean[$j] * $scale;
            }
            $cent[] = $row;
        }
        for ($i = 0; $i < $n; $i++) {
            $cent[$i][$i] = 0.0; // np.fill_diagonal(cent_distx, 0)
        }

        return [$cent, $rank];
    }

    /**
     * Column-wise dense ranks (1-based): rank[:, i] = dense_rank(distx[:, i]).
     *
     * @param list<list<float>> $distx
     *
     * @return list<list<int>>
     */
    private static function rankDistanceMatrix(array $distx): array
    {
        $n = count($distx);
        $rank = array_fill(0, $n, array_fill(0, $n, 0));
        for ($j = 0; $j < $n; $j++) {
            $col = [];
            for ($i = 0; $i < $n; $i++) {
                $col[$i] = $distx[$i][$j];
            }
            // dense rank via np.unique(return_inverse): sorted unique values -> 1-based index
            $uniq = array_values(array_unique($col, SORT_REGULAR));
            sort($uniq, SORT_NUMERIC);
            $idx = [];
            foreach ($uniq as $r => $v) {
                $idx[self::key($v)] = $r + 1;
            }
            for ($i = 0; $i < $n; $i++) {
                $rank[$i][$j] = $idx[self::key($col[$i])];
            }
        }

        return $rank;
    }

    private static function key(float $v): string
    {
        return sprintf('%.17g', $v);
    }

    /**
     * Port of _local_covariance (incl. _expected_covar and _covar_map).
     *
     * @param list<list<float>> $distx centered matrix
     * @param list<list<float>> $disty centered matrix (already transposed by caller)
     * @param list<list<int>>   $rankDistx 1-based ranks
     * @param list<list<int>>   $rankDisty 1-based ranks (already transposed by caller)
     *
     * @return list<list<float>>
     */
    private static function localCovariance(array $distx, array $disty, array $rankDistx, array $rankDisty): array
    {
        $n = count($distx);
        $nx = 0;
        $ny = 0;
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n; $j++) {
                if ($rankDistx[$i][$j] > $nx) {
                    $nx = $rankDistx[$i][$j];
                }
                if ($rankDisty[$i][$j] > $ny) {
                    $ny = $rankDisty[$i][$j];
                }
            }
        }
        // scipy: nx = max(rank-1)+1 = max(rank) since ranks are 1-based -> here nx already == max rank
        $cov = array_fill(0, $nx, array_fill(0, $ny, 0.0));
        $ex = array_fill(0, $nx, 0.0);
        $ey = array_fill(0, $ny, 0.0);
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $a = $distx[$i][$j];
                $b = $disty[$i][$j];
                $k = $rankDistx[$i][$j] - 1;
                $l = $rankDisty[$i][$j] - 1;
                $cov[$k][$l] += $a * $b;
                $ex[$k] += $a;
                $ey[$l] += $b;
            }
        }
        // cumulative sums along first column / first row
        for ($k = 1; $k < $nx; $k++) {
            $cov[$k][0] += $cov[$k - 1][0];
            $ex[$k] += $ex[$k - 1];
        }
        for ($l = 1; $l < $ny; $l++) {
            $cov[0][$l] += $cov[0][$l - 1];
            $ey[$l] += $ey[$l - 1];
        }
        // _covar_map: 2D cumulative via inclusion-exclusion
        for ($k = 0; $k < $nx - 1; $k++) {
            for ($l = 0; $l < $ny - 1; $l++) {
                $cov[$k + 1][$l + 1] += $cov[$k + 1][$l] + $cov[$k][$l + 1] - $cov[$k][$l];
            }
        }
        // subtract (ex outer ey) / n^2
        $n2 = (float) ($n * $n);
        for ($k = 0; $k < $nx; $k++) {
            for ($l = 0; $l < $ny; $l++) {
                $cov[$k][$l] -= ($ex[$k] * $ey[$l]) / $n2;
            }
        }

        return $cov;
    }

    /**
     * @param list<list<float>> $m
     *
     * @return list<float>
     */
    private static function diagonal(array $m): array
    {
        $out = [];
        $k = min(count($m), count($m[0]));
        for ($i = 0; $i < $k; $i++) {
            $out[] = $m[$i][$i];
        }

        return $out;
    }

    /**
     * @param list<list<float>> $m
     *
     * @return list<list<float>>
     */
    private static function transpose(array $m): array
    {
        $r = count($m);
        $c = count($m[0]);
        $t = array_fill(0, $c, array_fill(0, $r, 0.0));
        for ($i = 0; $i < $r; $i++) {
            for ($j = 0; $j < $c; $j++) {
                $t[$j][$i] = $m[$i][$j];
            }
        }

        return $t;
    }

    /**
     * Euclidean distance matrix cdist(x, x).
     *
     * @param list<list<float>> $x
     *
     * @return list<list<float>>
     */
    private static function euclideanDist(array $x): array
    {
        $n = count($x);
        $p = count($x[0]);
        $d = array_fill(0, $n, array_fill(0, $n, 0.0));
        for ($i = 0; $i < $n; $i++) {
            for ($j = $i + 1; $j < $n; $j++) {
                $s = 0.0;
                for ($k = 0; $k < $p; $k++) {
                    $diff = $x[$i][$k] - $x[$j][$k];
                    $s += $diff * $diff;
                }
                $v = sqrt($s);
                $d[$i][$j] = $v;
                $d[$j][$i] = $v;
            }
        }

        return $d;
    }

    /**
     * Apply symmetric permutation: permy = disty[order][:, order].
     *
     * @param list<list<float>> $disty
     * @param list<int>         $order
     *
     * @return list<list<float>>
     */
    private static function permuteSym(array $disty, array $order): array
    {
        $n = count($order);
        $out = array_fill(0, $n, array_fill(0, $n, 0.0));
        for ($i = 0; $i < $n; $i++) {
            $oi = $order[$i];
            for ($j = 0; $j < $n; $j++) {
                $out[$i][$j] = $disty[$oi][$order[$j]];
            }
        }

        return $out;
    }

    /**
     * @param list<list<float>> $u
     * @param list<list<float>> $v
     *
     * @return array{0: list<list<float>>, 1: list<list<float>>}
     */
    private static function twoSampleTransform(array $u, array $v): array
    {
        $nx = count($u);
        $ny = count($v);
        $x = array_merge($u, $v);
        $y = [];
        for ($i = 0; $i < $nx; $i++) {
            $y[] = [0.0];
        }
        for ($i = 0; $i < $ny; $i++) {
            $y[] = [1.0];
        }

        return [$x, $y];
    }

    /**
     * Coerce a 1-D list to (n,1), leave 2-D as is.
     *
     * @param list<float>|list<list<float>> $a
     *
     * @return list<list<float>>
     */
    private static function as2d(array $a): array
    {
        if (! is_array($a[0])) {
            $out = [];
            foreach ($a as $v) {
                $out[] = [(float) $v];
            }

            return $out;
        }
        $out = [];
        foreach ($a as $row) {
            $out[] = array_map('floatval', $row);
        }

        return $out;
    }
}
