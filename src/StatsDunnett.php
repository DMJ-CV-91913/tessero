<?php

declare(strict_types=1);

namespace Tessero;

use Tessero\Exceptions\TesseroException;
use Tessero\Random\Generator;

/**
 * scipy.stats.dunnett: Dunnett's single-step multiple-comparison test of several
 * experimental groups against one control group.
 *
 * This is a faithful, pure-PHP reimplementation of scipy.stats.dunnett (scipy 1.17.1,
 * scipy/stats/_multicomp.py) together with the multivariate-t machinery it relies on
 * (scipy/stats/_qmvnt.py + the `_qmvt_inner` Cython kernel). One class serves both
 * Tessero backends: all arithmetic is reimplemented here, the only external calls are to
 * Tessero\Special (ndtr / ndtri / gammaincinv) and Tessero\Random\Generator (NumPy-exact
 * PCG64 uniforms), both of which agree with NumPy/SciPy to machine precision.
 *
 * Exactness note
 * --------------
 * The `.statistic` is deterministic: it reproduces scipy to full double precision for
 * every input.
 *
 * The `.pvalue` is scipy's randomized-QMC estimate of the multivariate-t CDF over the
 * Dunnett correlation structure. Given the same `rng` seed it is reproducible, and this
 * implementation matches scipy bit-for-bit EXCEPT at a Fast-CBC-lattice construction step
 * that selects a lattice generator via `argmin` of an FFT-based circular convolution. For
 * a subset of group counts the minimiser is a numerical tie (two candidates equal to
 * ~1e-16), and scipy's choice there is decided by pocketfft's exact rounding -- which even
 * differs between numpy.fft and scipy.fft. For those group counts the p-value is therefore
 * not portable to full precision (it drifts at the QMC-noise level, ~1e-4). This method
 * reproduces the lattice via an exact O(m^2) circular convolution, so it agrees with scipy
 * to full precision for the "tie-free" group counts and is a statistically-equivalent QMC
 * estimate otherwise. The parity fixtures exercise `.pvalue` only on the tie-free group
 * counts; `.statistic` is exercised for all.
 *
 * @see \Tessero\Stats::tukeyHsd for all-pairs comparison (Tukey HSD).
 */
final class StatsDunnett
{
    /**
     * scipy.stats.dunnett(*samples, control=..., alternative=..., rng=...).
     *
     * @param list<list<float>> $samples     one 1-D sample per experimental group
     * @param list<float>       $control     the control-group sample
     * @param string            $alternative 'two-sided' (default), 'less' or 'greater'
     * @param int|array<int>|null $seed      seed for NumPy-exact PCG64 (scipy's `rng`); null => os entropy
     *
     * @return array{statistic: list<float>, pvalue: list<float>}
     */
    public static function dunnett(
        array $samples,
        array $control,
        string $alternative = 'two-sided',
        int|array|null $seed = null
    ): array {
        if (! in_array($alternative, ['two-sided', 'less', 'greater'], true)) {
            throw new TesseroException("dunnett: alternative must be 'less', 'greater' or 'two-sided'.");
        }
        if (count($samples) < 1) {
            throw new TesseroException('dunnett: at least one sample group is required.');
        }

        $groups = [];
        foreach ($samples as $s) {
            $g = array_map('floatval', array_values((array) $s));
            if (count($g) < 1) {
                throw new TesseroException('dunnett: control and samples groups must have at least 1 observation.');
            }
            $groups[] = $g;
        }
        $ctrl = array_map('floatval', array_values($control));
        if (count($ctrl) < 1) {
            throw new TesseroException('dunnett: control and samples groups must have at least 1 observation.');
        }

        $nGroups = count($groups);
        $nSamples = array_map('count', $groups);
        $nSample = array_sum($nSamples);
        $nControl = count($ctrl);
        $n = $nSample + $nControl;
        $df = $n - $nGroups - 1;

        // rho_ij = 1/sqrt((N0/Ni + 1)(N0/Nj + 1)); diagonal = 1. [Dunnett 1955, p.1103]
        $rhoDiag = [];
        for ($i = 0; $i < $nGroups; $i++) {
            $rhoDiag[$i] = $nControl / $nSamples[$i] + 1.0;
        }
        $rho = [];
        for ($i = 0; $i < $nGroups; $i++) {
            $rho[$i] = [];
            for ($j = 0; $j < $nGroups; $j++) {
                $rho[$i][$j] = ($i === $j) ? 1.0 : 1.0 / sqrt($rhoDiag[$i] * $rhoDiag[$j]);
            }
        }

        // statistic: standardized mean differences vs control.
        $meanControl = array_sum($ctrl) / $nControl;
        $meanSamples = [];
        foreach ($groups as $g) {
            $meanSamples[] = array_sum($g) / count($g);
        }
        // pooled variance estimate s^2 [Dunnett 1955, Eq.1]
        $ss = self::sumSq($ctrl, $meanControl);
        foreach ($groups as $gi => $g) {
            $ss += self::sumSq($g, $meanSamples[$gi]);
        }
        $s2 = $ss / $df;
        $std = sqrt($s2);

        $statistic = [];
        for ($i = 0; $i < $nGroups; $i++) {
            $z = ($meanSamples[$i] - $meanControl) / sqrt(1.0 / $nSamples[$i] + 1.0 / $nControl);
            $statistic[$i] = $z / $std;
        }

        $pvalue = self::pvalue($rho, $df, $statistic, $alternative, $seed);

        return ['statistic' => $statistic, 'pvalue' => $pvalue];
    }

    /** @param list<float> $x */
    private static function sumSq(array $x, float $mean): float
    {
        $acc = 0.0;
        foreach ($x as $v) {
            $d = $v - $mean;
            $acc += $d * $d;
        }
        return $acc;
    }

    /**
     * p-value from the multivariate-t distribution (mirrors _pvalue_dunnett + multivariate_t._cdf).
     *
     * @param list<list<float>> $rho
     * @param list<float>       $statistic
     * @return list<float>
     */
    private static function pvalue(array $rho, int $df, array $statistic, string $alternative, int|array|null $seed): array
    {
        $dim = count($statistic);
        $maxpts = 1000 * $dim;
        $rng = Generator::defaultRng($seed);

        // Each comparison i -> a full `dim`-dimensional mvt-CDF call (mirrors scipy's
        // `statistic.reshape(-1,1)` broadcast against loc=zeros(dim), then apply_along_axis).
        $out = [];
        for ($i = 0; $i < $dim; $i++) {
            $si = $statistic[$i];
            if ($alternative === 'two-sided') {
                $xi = abs($si);
                $a = array_fill(0, $dim, -$xi);
                $b = array_fill(0, $dim, $xi);
            } elseif ($alternative === 'greater') {
                $a = array_fill(0, $dim, -INF);
                $b = array_fill(0, $dim, $si);
            } else { // less
                $a = array_fill(0, $dim, $si);
                $b = array_fill(0, $dim, INF);
            }

            // swap bounds where b < a, tracking the sign (mirrors _cdf i_swap/signs).
            $sign = 1.0;
            for ($k = 0; $k < $dim; $k++) {
                if ($b[$k] < $a[$k]) {
                    $t = $a[$k];
                    $a[$k] = $b[$k];
                    $b[$k] = $t;
                    $sign = -$sign;
                }
            }
            $cdf = self::qmvt($maxpts, $df, $rho, $a, $b, $rng) * $sign;
            $out[$i] = 1.0 - $cdf;
        }
        return $out;
    }

    /**
     * Multivariate-t box integral by randomized QMC (mirrors scipy _qmvt).
     *
     * @param list<list<float>> $covar
     * @param list<float>       $low
     * @param list<float>       $high
     */
    private static function qmvt(int $m, int $nu, array $covar, array $low, array $high, Generator $rng): float
    {
        $sn = max(1.0, sqrt((float) $nu));
        $n = count($covar);
        $lowS = [];
        $highS = [];
        for ($i = 0; $i < $n; $i++) {
            $lowS[$i] = $low[$i] / $sn;
            $highS[$i] = $high[$i] / $sn;
        }
        [$cho, $lo, $hi] = self::permutedCholesky($covar, $lowS, $highS);

        [$q, $nQmc] = self::cbcLattice($n, intdiv(max($m, 1), 10) > 0 ? intdiv($m, 10) : 1);
        $nBatches = 10;
        // rng.random(size=(n_batches, n)) -- NumPy-exact PCG64 stream.
        $rndmNd = $rng->random([$nBatches, $n]);
        $rndm = method_exists($rndmNd, 'toArray') ? $rndmNd->toArray() : $rndmNd;

        return self::qmvtInner($q, $rndm, $nQmc, $nBatches, $cho, $lo, $hi, (float) $nu);
    }

    /**
     * The `_qmvt_inner` kernel (scipy _qmvnt_cy.pyx), pure PHP.
     *
     * @param list<float>         $q
     * @param list<list<float>>   $rndm  shape (n_batches, n)
     * @param list<list<float>>   $cho
     * @param list<float>         $lo
     * @param list<float>         $hi
     */
    private static function qmvtInner(array $q, array $rndm, int $nQmc, int $nBatches, array $cho, array $lo, array $hi, float $nu): float
    {
        $n = count($cho);
        $prob = 0.0;
        $errorVar = 0.0;

        for ($j = 0; $j < $nBatches; $j++) {
            $pv = array_fill(0, $nQmc, 1.0);
            // s[i][k]
            $s = [];
            for ($i = 0; $i < $n; $i++) {
                $s[$i] = array_fill(0, $nQmc, 0.0);
            }

            // r_k = sqrt(2 * gammaincinv(nu/2, x_k)) ;  x from dim 0 of the lattice.
            $r = array_fill(0, $nQmc, 1.0);
            if ($nu > 0) {
                $xArg = [];
                $q0 = $q[0];
                $r0 = $rndm[$j][0];
                for ($k = 0; $k < $nQmc; $k++) {
                    $z = $q0 * ($k + 1) + $r0;
                    $z -= (float) (int) $z; // z - floor via trunc of positive value
                    $xArg[$k] = abs(2.0 * $z - 1.0);
                }
                $g = Special::gammaincinv($nu / 2.0, $xArg);
                $gArr = method_exists($g, 'toArray') ? $g->toArray() : (is_array($g) ? $g : [$g]);
                for ($k = 0; $k < $nQmc; $k++) {
                    $r[$k] = sqrt(2.0 * $gArr[$k]);
                }
            }

            $c = array_fill(0, $nQmc, 1.0);
            $dc = array_fill(0, $nQmc, 0.0);

            for ($i = 0; $i < $n; $i++) {
                $qq = $q[$i];
                $rr = $rndm[$j][$i];

                if ($i > 0) {
                    // y_k = ndtri(c_k + x_k*dc_k) ; accumulate into s[ip][k]
                    $arg = [];
                    for ($k = 0; $k < $nQmc; $k++) {
                        $zk = $qq * ($k + 1) + $rr;
                        $zk -= (float) (int) $zk;
                        $xk = abs(2.0 * $zk - 1.0);
                        $arg[$k] = $c[$k] + $xk * $dc[$k];
                    }
                    $yNd = Special::ndtri($arg);
                    $y = method_exists($yNd, 'toArray') ? $yNd->toArray() : (is_array($yNd) ? $yNd : [$yNd]);
                    for ($ip = $i; $ip < $n; $ip++) {
                        $choIp = $cho[$ip][$i - 1];
                        if ($choIp === 0.0) {
                            continue;
                        }
                        $sip = &$s[$ip];
                        for ($k = 0; $k < $nQmc; $k++) {
                            $sip[$k] += $choIp * $y[$k];
                        }
                        unset($sip);
                    }
                }

                $loi = $lo[$i];
                $hii = $hi[$i];
                $si = $s[$i];
                for ($k = 0; $k < $nQmc; $k++) {
                    $lois = $loi * $r[$k] - $si[$k];
                    if ($lois < -9.0) {
                        $ck = 0.0;
                    } elseif ($lois < 9.0) {
                        $ck = self::phi($lois);
                    } else {
                        $ck = 1.0;
                    }
                    $hiis = $hii * $r[$k] - $si[$k];
                    if ($hiis < -9.0) {
                        $dk = 0.0;
                    } elseif ($hiis < 9.0) {
                        $dk = self::phi($hiis);
                    } else {
                        $dk = 1.0;
                    }
                    $c[$k] = $ck;
                    $dcv = $dk - $ck;
                    $dc[$k] = $dcv;
                    $pv[$k] *= $dcv;
                }
            }

            $mean = array_sum($pv) / $nQmc;
            $dd = ($mean - $prob) / ($j + 1);
            $prob += $dd;
            $errorVar = ($j - 1) * $errorVar / ($j + 1) + $dd * $dd;
        }

        return $prob;
    }

    /** Standard normal CDF (scipy ndtr), scalar. */
    private static function phi(float $x): float
    {
        return (float) Special::ndtr($x);
    }

    /**
     * Scaled, permuted Cholesky factor with transformed bounds (scipy _permuted_cholesky).
     *
     * @param list<list<float>> $covar
     * @param list<float>       $low
     * @param list<float>       $high
     * @return array{0: list<list<float>>, 1: list<float>, 2: list<float>}
     */
    private static function permutedCholesky(array $covar, array $low, array $high, float $tol = 1e-10): array
    {
        $n = count($covar);
        $cho = [];
        for ($i = 0; $i < $n; $i++) {
            $cho[$i] = array_values($covar[$i]);
        }
        $lo = array_values($low);
        $hi = array_values($high);

        // scale by sqrt(diag), guarding zeros.
        $dc = [];
        for ($i = 0; $i < $n; $i++) {
            $d = sqrt(max($cho[$i][$i], 0.0));
            $dc[$i] = ($d === 0.0) ? 1.0 : $d;
        }
        for ($i = 0; $i < $n; $i++) {
            $lo[$i] /= $dc[$i];
            $hi[$i] /= $dc[$i];
        }
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $cho[$i][$j] = $cho[$i][$j] / $dc[$j] / $dc[$i];
            }
        }

        $y = array_fill(0, $n, 0.0);
        $sqtp = sqrt(2.0 * M_PI);

        for ($k = 0; $k < $n; $k++) {
            $epk = ($k + 1) * $tol;
            $im = $k;
            $ck = 0.0;
            $dem = 1.0;
            $loM = 0.0;
            $hiM = 0.0;
            for ($i = $k; $i < $n; $i++) {
                if ($cho[$i][$i] > $tol) {
                    $ci = sqrt($cho[$i][$i]);
                    $sdot = 0.0;
                    if ($i > 0) {
                        for ($t = 0; $t < $k; $t++) {
                            $sdot += $cho[$i][$t] * $y[$t];
                        }
                    }
                    $loI = ($lo[$i] - $sdot) / $ci;
                    $hiI = ($hi[$i] - $sdot) / $ci;
                    $de = self::phi($hiI) - self::phi($loI);
                    if ($de <= $dem) {
                        $ck = $ci;
                        $dem = $de;
                        $loM = $loI;
                        $hiM = $hiI;
                        $im = $i;
                    }
                }
            }

            if ($im > $k) {
                // partial swap of rows/cols im and k (mirrors scipy's _swap_slices block).
                $cho[$im][$im] = $cho[$k][$k];
                for ($t = 0; $t < $k; $t++) {
                    $tmp = $cho[$im][$t];
                    $cho[$im][$t] = $cho[$k][$t];
                    $cho[$k][$t] = $tmp;
                }
                for ($t = $im + 1; $t < $n; $t++) {
                    $tmp = $cho[$t][$im];
                    $cho[$t][$im] = $cho[$t][$k];
                    $cho[$t][$k] = $tmp;
                }
                for ($t = $k + 1; $t < $im; $t++) {
                    // swap cho[t][k] with cho[im][t]
                    $tmp = $cho[$t][$k];
                    $cho[$t][$k] = $cho[$im][$t];
                    $cho[$im][$t] = $tmp;
                }
                $tmp = $lo[$k];
                $lo[$k] = $lo[$im];
                $lo[$im] = $tmp;
                $tmp = $hi[$k];
                $hi[$k] = $hi[$im];
                $hi[$im] = $tmp;
            }

            if ($ck > $epk) {
                $cho[$k][$k] = $ck;
                for ($t = $k + 1; $t < $n; $t++) {
                    $cho[$k][$t] = 0.0;
                }
                for ($i = $k + 1; $i < $n; $i++) {
                    $cho[$i][$k] /= $ck;
                    // cho[i, k+1:i+1] -= cho[i,k] * cho[k+1:i+1, k]
                    $cik = $cho[$i][$k];
                    for ($t = $k + 1; $t <= $i; $t++) {
                        $cho[$i][$t] -= $cik * $cho[$t][$k];
                    }
                }
                if (abs($dem) > $tol) {
                    $y[$k] = (exp(-$loM * $loM / 2.0) - exp(-$hiM * $hiM / 2.0)) / ($sqtp * $dem);
                } else {
                    $y[$k] = ($loM + $hiM) / 2.0;
                    if ($loM < -10.0) {
                        $y[$k] = $hiM;
                    } elseif ($hiM > 10.0) {
                        $y[$k] = $loM;
                    }
                }
                // cho[k, :k+1] /= ck ; lo[k] /= ck ; hi[k] /= ck
                for ($t = 0; $t <= $k; $t++) {
                    $cho[$k][$t] /= $ck;
                }
                $lo[$k] /= $ck;
                $hi[$k] /= $ck;
            } else {
                for ($i = $k; $i < $n; $i++) {
                    $cho[$i][$k] = 0.0;
                }
                $y[$k] = ($lo[$k] + $hi[$k]) / 2.0;
            }
        }

        return [$cho, $lo, $hi];
    }

    /**
     * Fast CBC lattice generator (scipy _cbc_lattice). The FFT-based circular convolution
     * `ifft(fft(c)*fft(q)).real` is computed here by an exact O(m^2) circular convolution,
     * which agrees with scipy to full precision except at argmin ties (see class docstring).
     *
     * @return array{0: list<float>, 1: int}
     */
    private static function cbcLattice(int $nDim, int $nQmcSamples): array
    {
        $primes = self::primesFrom2To($nQmcSamples + 1);
        $nQmc = (int) end($primes);

        $m = intdiv($nQmc - 1, 2);
        $g = self::primitiveRoot($nQmc);

        $perm = array_fill(0, $m, 1);
        for ($j = 0; $j < $m - 1; $j++) {
            $perm[$j + 1] = ($g * $perm[$j]) % $nQmc;
        }
        for ($j = 0; $j < $m; $j++) {
            $perm[$j] = min($nQmc - $perm[$j], $perm[$j]);
        }
        $c = [];
        for ($j = 0; $j < $m; $j++) {
            $p = $perm[$j] / $nQmc;
            $c[$j] = $p * $p - $p + 1.0 / 6.0;
        }

        // bt=ones ; gm = [1, 0.8^0, 0.8^1, ...]
        $gm = [1.0];
        for ($i = 0; $i < $nDim - 1; $i++) {
            $gm[] = 0.8 ** $i;
        }

        // scipy starts q as the scalar 1 and broadcasts against `reordered` (length m)
        // on the first multiply; keep q as a length-m vector of ones.
        $q = array_fill(0, $m, 1.0);
        $w = 0;
        $z = range(1, $nDim); // z[0..nDim-1] = 1..nDim

        for ($s = 1; $s < $nDim; $s++) {
            // reordered = hstack([c[:w+1][::-1], c[w+1:m][::-1]])
            $reordered = [];
            for ($t = $w; $t >= 0; $t--) {
                $reordered[] = $c[$t];
            }
            for ($t = $m - 1; $t > $w; $t--) {
                $reordered[] = $c[$t];
            }
            // q = q * (bt[s-1] + gm[s-1]*reordered)   (bt=1)
            $gs = $gm[$s - 1];
            $mlen = count($q);
            for ($t = 0; $t < $mlen; $t++) {
                $q[$t] = $q[$t] * (1.0 + $gs * $reordered[$t]);
            }
            // conv[t] = sum_j c[j]*q[(t-j) mod mlen] ; w = argmin(conv)
            $bestT = 0;
            $bestV = INF;
            for ($t = 0; $t < $mlen; $t++) {
                $acc = 0.0;
                for ($jj = 0; $jj < $mlen; $jj++) {
                    $idx = ($t - $jj) % $mlen;
                    if ($idx < 0) {
                        $idx += $mlen;
                    }
                    $acc += $c[$jj] * $q[$idx];
                }
                if ($acc < $bestV) {
                    $bestV = $acc;
                    $bestT = $t;
                }
            }
            $w = $bestT;
            $z[$s] = $perm[$w];
        }

        $qOut = [];
        for ($i = 0; $i < $nDim; $i++) {
            $qOut[$i] = $z[$i] / $nQmc;
        }
        return [$qOut, $nQmc];
    }

    /**
     * Primes in [2, n) via a simple sieve (mirrors scipy primes_from_2_to usage).
     *
     * @return list<int>
     */
    private static function primesFrom2To(int $n): array
    {
        if ($n < 3) {
            return $n === 2 ? [2] : [];
        }
        $sieve = array_fill(0, intdiv($n, 2), true); // sieve[i] represents 2i+1
        for ($i = 3; $i * $i < $n; $i += 2) {
            if ($sieve[intdiv($i, 2)]) {
                for ($k = intdiv($i * $i, 2); $k < intdiv($n, 2); $k += $i) {
                    $sieve[$k] = false;
                }
            }
        }
        $out = [2];
        for ($i = 1; $i < intdiv($n, 2); $i++) {
            if ($sieve[$i]) {
                $out[] = 2 * $i + 1;
            }
        }
        return $out;
    }

    /** Smallest primitive root mod prime p (scipy _primitive_root). */
    private static function primitiveRoot(int $p): int
    {
        $pm = $p - 1;
        $factors = self::factorize($pm);
        $nf = count($factors);
        $r = 2;
        $k = 0;
        while ($k < $nf) {
            $d = intdiv($pm, $factors[$k]);
            if (self::powmod($r, $d, $p) === 1) {
                $r++;
                $k = 0;
            } else {
                $k++;
            }
        }
        return $r;
    }

    /** @return list<int> sorted unique prime factors */
    private static function factorize(int $n): array
    {
        $factors = [];
        foreach (self::primesFrom2To((int) sqrt($n) + 1) as $p) {
            while ($n % $p === 0) {
                $factors[$p] = true;
                $n = intdiv($n, $p);
            }
            if ($n === 1) {
                break;
            }
        }
        if ($n !== 1) {
            $factors[$n] = true;
        }
        $keys = array_keys($factors);
        sort($keys);
        return $keys;
    }

    /** (base^exp) mod mod, with exp >= 0, using PHP ints (p < 2^31 so products fit in 64-bit). */
    private static function powmod(int $base, int $exp, int $mod): int
    {
        $result = 1 % $mod;
        $base %= $mod;
        while ($exp > 0) {
            if (($exp & 1) === 1) {
                $result = ($result * $base) % $mod;
            }
            $exp >>= 1;
            $base = ($base * $base) % $mod;
        }
        return $result;
    }
}
