<?php

declare(strict_types=1);

namespace Tessero;

use Tessero\Exceptions\TesseroException;

/**
 * scipy.stats.logrank: two-sample log-rank test comparing the survival distributions of two
 * right-censored samples via their empirical survival functions.
 *
 * Pure PHP, array-in / array-out, deterministic (no RNG, no kernel call) -- mirrors the
 * {@see StatsFns} style: all arithmetic is reimplemented here so one class serves both backends.
 *
 * Inputs mirror scipy.stats.CensoredData(uncensored=..., right=...): each sample is given as two
 * lists -- the uncensored (observed-event / "death") times and the right-censored times. (Tessero has
 * no CensoredData object; the two lists ARE the CensoredData, matching its `_uncensored` / `_right`.)
 *
 * The statistic is the signed standard-normal Z of references (scipy's convention; its square is the
 * chi-square(df=1) statistic reported by most other implementations). The p-value uses the standard
 * normal survival/CDF; for the default 'two-sided' alternative, pvalue = 2*Phi_sf(|Z|), which equals
 * chi2(df=1).sf(Z**2). Tessero also exposes the chi-square route (special.gammaincc(0.5, Z**2/2)); the
 * normal form is used here because it also serves the one-sided alternatives and needs no kernel call.
 */
final class LogRank
{
    /**
     * scipy.stats.logrank(x, y, alternative).
     *
     * @param list<float> $xUncensored observed-event times of sample X (CensoredData.uncensored)
     * @param list<float> $xRight       right-censored times of sample X (CensoredData.right)
     * @param list<float> $yUncensored observed-event times of sample Y
     * @param list<float> $yRight       right-censored times of sample Y
     * @param string      $alternative  'two-sided' (default), 'less' or 'greater'
     *
     * @return array{statistic: float, pvalue: float}
     */
    public static function logrank(
        array $xUncensored,
        array $xRight = [],
        array $yUncensored = [],
        array $yRight = [],
        string $alternative = 'two-sided'
    ): array {
        $xu = array_map('floatval', array_values($xUncensored));
        $xr = array_map('floatval', array_values($xRight));
        $yu = array_map('floatval', array_values($yUncensored));
        $yr = array_map('floatval', array_values($yRight));
        foreach ([$xu, $xr, $yu, $yr] as $arr) {
            foreach ($arr as $v) {
                if (! is_finite($v)) {
                    throw new TesseroException('logrank: all times must be finite.');
                }
            }
        }
        if (! in_array($alternative, ['two-sided', 'less', 'greater'], true)) {
            throw new TesseroException("logrank: alternative must be 'two-sided', 'less', or 'greater'.");
        }

        // All observed times (event or censored), sorted ascending, per sample and combined.
        $xAll = array_merge($xu, $xr);
        $yAll = array_merge($yu, $yr);
        $xyAll = array_merge($xAll, $yAll);
        sort($xAll);
        sort($xyAll);

        // Combined death counts per unique time (only event times contribute).
        // deaths_xy[t] = number of uncensored (event) observations in the combined sample at time t.
        $deaths = [];                                  // map time -> combined event count
        foreach (array_merge($xu, $yu) as $t) {
            $k = self::key($t);
            $deaths[$k] = ($deaths[$k] ?? 0) + 1;
        }
        // Unique event times in ascending order.
        $eventTimes = [];
        foreach ($deaths as $k => $_) {
            $eventTimes[] = (float) $k;
        }
        sort($eventTimes);

        $sumVar = 0.0;         // sum_j V_{x,j}
        $sumExpDeathsX = 0.0;  // sum_j E_{x,j}
        foreach ($eventTimes as $t) {
            $d = $deaths[self::key($t)];               // O_j: combined observed events at t
            $Nj = self::atRisk($xyAll, $t);            // N_j: combined at risk just before t
            $Nxj = self::atRisk($xAll, $t);            // N_{x,j}: at risk in X just before t
            $Nyj = $Nj - $Nxj;                         // N_{y,j}
            // E_{x,j} = O_j * N_{x,j} / N_j
            $sumExpDeathsX += $Nxj * ($d / $Nj);
            // V_{x,j} = N_{x,j} N_{y,j} O_j (N_j - O_j) / (N_j^2 (N_j - 1)); 0 when N_j == 1.
            if ($Nj > 1) {
                $num = $Nxj * $Nyj * $d * ($Nj - $d);
                $den = $Nj * $Nj * ($Nj - 1);
                $sumVar += $num / $den;
            }
        }

        $nDiedX = count($xu);                          // O_x: total observed events in X
        if ($sumVar <= 0.0) {
            throw new TesseroException('logrank: degenerate samples (zero variance).');
        }
        $statistic = ($nDiedX - $sumExpDeathsX) / sqrt($sumVar);

        $pvalue = match ($alternative) {
            'less'    => self::normCdf($statistic),
            'greater' => self::normSf($statistic),
            default   => 2.0 * self::normSf(abs($statistic)),   // two-sided (symmetric)
        };

        return ['statistic' => $statistic, 'pvalue' => $pvalue];
    }

    /** Number at risk just before time $t = count of observations with time >= $t (array sorted ascending). */
    private static function atRisk(array $sortedAsc, float $t): int
    {
        // first index with value >= t  (bisect_left)
        $lo = 0;
        $hi = count($sortedAsc);
        while ($lo < $hi) {
            $mid = ($lo + $hi) >> 1;
            if ($sortedAsc[$mid] < $t) {
                $lo = $mid + 1;
            } else {
                $hi = $mid;
            }
        }
        return count($sortedAsc) - $lo;
    }

    /** Stable string key for a float time (collapses -0.0/+0.0; full round-trip precision preserved). */
    private static function key(float $t): string
    {
        return sprintf('%.17g', $t + 0.0);
    }

    /** Standard-normal survival function 1-Phi(z) = ndtr(-z). */
    private static function normSf(float $z): float
    {
        return self::ndtr(-$z);
    }

    /** Standard-normal CDF Phi(z) = ndtr(z). */
    private static function normCdf(float $z): float
    {
        return self::ndtr($z);
    }

    /**
     * Standard-normal CDF (Cephes `ndtr`, the routine scipy.special.ndtr wraps); full double precision.
     */
    private static function ndtr(float $a): float
    {
        $sqrth = 0.70710678118654752440; // 1/sqrt(2)
        $x = $a * $sqrth;
        $z = abs($x);
        if ($z < $sqrth) {
            return 0.5 + 0.5 * self::erf($x);
        }
        $y = 0.5 * self::erfc($z);
        return $x > 0.0 ? 1.0 - $y : $y;
    }

    /** Horner evaluation of a polynomial with leading coefficient $c[0]. */
    private static function polevl(float $x, array $c): float
    {
        $r = $c[0];
        for ($i = 1, $n = count($c); $i < $n; $i++) {
            $r = $r * $x + $c[$i];
        }
        return $r;
    }

    /** Horner evaluation of a monic polynomial (implicit leading coefficient 1). */
    private static function p1evl(float $x, array $c): float
    {
        $r = $x + $c[0];
        for ($i = 1, $n = count($c); $i < $n; $i++) {
            $r = $r * $x + $c[$i];
        }
        return $r;
    }

    /** Error function erf(x) -- Cephes rational approximation (|x| <= 1); full double precision. */
    private static function erf(float $x): float
    {
        static $T = [
            9.60497373987051638749e0, 9.00260197203842689217e1,
            2.23200534594684319226e3, 7.00332514112805075473e3,
            5.55923013010394962768e4,
        ];
        static $U = [
            3.35617141647503099647e1, 5.21357949780152679795e2,
            4.59432382970980127987e3, 2.26290000613890934246e4,
            4.92673942608635921086e4,
        ];
        if (abs($x) > 1.0) {
            return 1.0 - self::erfc($x);
        }
        $z = $x * $x;
        return $x * self::polevl($z, $T) / self::p1evl($z, $U);
    }

    /** Complementary error function erfc(x) -- Cephes rational approximation; full double precision. */
    private static function erfc(float $a): float
    {
        static $P = [
            2.46196981473530512524e-10, 5.64189564831068821977e-1,
            7.46321056442269912687e0, 4.86371970985681366614e1,
            1.96520832956077098242e2, 5.26445194995477358631e2,
            9.34528527171957607540e2, 1.02755188689515710272e3,
            5.57535335369399327526e2,
        ];
        static $Q = [
            1.32281951154744992508e1, 8.67072140885989742329e1,
            3.54937778887819891062e2, 9.75708501743205489753e2,
            1.82390916687909736289e3, 2.24633760818710981792e3,
            1.65666309194161350182e3, 5.57535340817727675546e2,
        ];
        static $R = [
            5.64189583547755073984e-1, 1.27536670759978104416e0,
            5.01905042251180477414e0, 6.16021097993053585195e0,
            7.40974269950448939160e0, 2.97886665372100240670e0,
        ];
        static $S = [
            2.26052863220117276590e0, 9.39603524938001434673e0,
            1.20489539808096656605e1, 1.70814450747565897222e1,
            9.60896809063285878198e0, 3.36907645100081516050e0,
        ];
        $x = abs($a);
        if ($x < 1.0) {
            return 1.0 - self::erf($a);
        }
        $z = -$a * $a;
        if ($z < -708.396418532264078748) {
            return $a < 0.0 ? 2.0 : 0.0;
        }
        $z = exp($z);
        if ($x < 8.0) {
            $p = self::polevl($x, $P);
            $q = self::p1evl($x, $Q);
        } else {
            $p = self::polevl($x, $R);
            $q = self::p1evl($x, $S);
        }
        $y = ($z * $p) / $q;
        if ($a < 0.0) {
            $y = 2.0 - $y;
        }
        if ($y === 0.0) {
            return $a < 0.0 ? 2.0 : 0.0;
        }
        return $y;
    }
}
