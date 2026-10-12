<?php

declare(strict_types=1);

namespace Tessero;

use Tessero\Exceptions\TesseroException;
use Tessero\Optimize\Minimize;

/**
 * scipy.stats helpers that select an optimal transform parameter by minimising a 1-D objective over the
 * data: {@see boxcoxNormmax} and {@see yeojohnsonNormmax}. These are pure PHP (the Box-Cox / Yeo-Johnson
 * transforms, their log-likelihoods, Pearson's r and the inverse normal CDF are reimplemented here) driving
 * Tessero's scipy-matched Brent / bounded minimisers, so one class serves both backends with no kernel call.
 */
final class StatsFns
{
    /**
     * scipy.stats.boxcox_normmax: the Box-Cox parameter lambda that best normalises x.
     *
     * @param list<float> $x positive data
     */
    public static function boxcoxNormmax(array $x, string $method = 'pearsonr'): float
    {
        $x = array_map('floatval', array_values($x));
        foreach ($x as $v) {
            if (! ($v > 0) || ! is_finite($v)) {
                throw new TesseroException('boxcox_normmax: x must be positive and finite.');
            }
        }
        if ($method === 'mle') {
            return Minimize::brent(fn (float $l): float => -self::boxcoxLlf($l, $x), [-2.0, 2.0]);
        }
        if ($method === 'pearsonr') {
            $n = count($x);
            $xvals = array_map([self::class, 'ndtri'], self::uom($n));
            $obj = static function (float $lmb) use ($x, $xvals): float {
                $y = self::boxcox($x, $lmb);
                sort($y);
                return 1.0 - self::pearsonr($xvals, $y);
            };
            return Minimize::brent($obj, [-2.0, 2.0]);
        }
        throw new TesseroException("boxcox_normmax: unsupported method '{$method}'.");
    }

    /**
     * scipy.stats.yeojohnson_normmax: the Yeo-Johnson parameter lambda maximising the log-likelihood.
     *
     * @param list<float> $x
     */
    public static function yeojohnsonNormmax(array $x): float
    {
        $x = array_map('floatval', array_values($x));
        foreach ($x as $v) {
            if (! is_finite($v)) {
                throw new TesseroException('yeojohnson_normmax: x must be finite.');
            }
        }
        $allZero = true;
        $maxabs = 0.0;
        $anyNeg = false;
        $allNeg = true;
        foreach ($x as $v) {
            if ($v !== 0.0) { $allZero = false; }
            if ($v < 0) { $anyNeg = true; } else { $allNeg = false; }
            $maxabs = max($maxabs, abs($v));
        }
        if ($allZero) {
            return 1.0;
        }
        $log1pMax = log1p(20.0 * $maxabs);
        $logEps = log(2.220446049250313e-16);
        $logTiny = (log(2.2250738585072014e-308) - $logEps) / 2.0;
        $logMax = (log(1.7976931348623157e+308) + $logEps) / 2.0;
        $lb = $logTiny / $log1pMax;
        $ub = $logMax / $log1pMax;
        if ($allNeg) {
            [$lb, $ub] = [2.0 - $ub, 2.0 - $lb];
        } elseif ($anyNeg) {
            [$lb, $ub] = [max(2.0 - $ub, $lb), min(2.0 - $lb, $ub)];
        }
        return Minimize::fminbound(fn (float $l): float => -self::yjLlf($l, $x), $lb, $ub, 1.48e-8);
    }

    /**
     * scipy.stats.ppcc_max: the shape parameter (Tukey-lambda) maximising the probability-plot
     * correlation coefficient for x.
     *
     * @param list<float> $x
     */
    public static function ppccMax(array $x): float
    {
        $x = array_map('floatval', array_values($x));
        foreach ($x as $v) {
            if (! is_finite($v)) {
                throw new TesseroException('ppcc_max: x must be finite.');
            }
        }
        sort($x);
        $mi = self::uom(count($x));
        $obj = static function (float $shape) use ($mi, $x): float {
            $q = [];
            foreach ($mi as $i => $m) {
                $q[$i] = self::tukeyPpf($m, $shape);
            }
            return 1.0 - self::pearsonr($q, $x);
        };
        return Minimize::brent($obj, [0.0, 1.0]);
    }

    // ---- pure-PHP helpers ----------------------------------------------------------------

    /** Tukey-lambda quantile function. */
    private static function tukeyPpf(float $q, float $lmb): float
    {
        return $lmb === 0.0 ? log($q / (1.0 - $q)) : ($q ** $lmb - (1.0 - $q) ** $lmb) / $lmb;
    }

    /** @param list<float> $x @return list<float> */
    private static function boxcox(array $x, float $lmb): array
    {
        $out = [];
        if ($lmb === 0.0) {
            foreach ($x as $v) { $out[] = log($v); }
        } else {
            foreach ($x as $v) { $out[] = ($v ** $lmb - 1.0) / $lmb; }
        }
        return $out;
    }

    /** @param list<float> $x */
    private static function boxcoxLlf(float $lmb, array $x): float
    {
        $n = count($x);
        $sumlog = 0.0;
        foreach ($x as $v) { $sumlog += log($v); }
        $y = self::boxcox($x, $lmb);
        return ($lmb - 1.0) * $sumlog - $n / 2.0 * log(self::popvar($y));
    }

    /** Yeo-Johnson transform. @param list<float> $x @return list<float> */
    private static function yj(array $x, float $lmb): array
    {
        $out = [];
        foreach ($x as $v) {
            if ($v >= 0) {
                $out[] = $lmb !== 0.0 ? (($v + 1.0) ** $lmb - 1.0) / $lmb : log1p($v);
            } else {
                $out[] = $lmb !== 2.0 ? -(((-$v + 1.0) ** (2.0 - $lmb)) - 1.0) / (2.0 - $lmb) : -log1p(-$v);
            }
        }
        return $out;
    }

    /** @param list<float> $x */
    private static function yjLlf(float $lmb, array $x): float
    {
        $n = count($x);
        $trans = self::yj($x, $lmb);
        $ll = -$n / 2.0 * log(self::popvar($trans));
        $s = 0.0;
        foreach ($x as $v) { $s += ($v < 0 ? -1.0 : 1.0) * log1p(abs($v)); }
        return $ll + ($lmb - 1.0) * $s;
    }

    /** @param list<float> $a */
    private static function popvar(array $a): float
    {
        $n = count($a);
        $m = 0.0;
        foreach ($a as $v) { $m += $v; }
        $m /= $n;
        $s = 0.0;
        foreach ($a as $v) { $s += ($v - $m) * ($v - $m); }
        return $s / $n;
    }

    /** @param list<float> $a @param list<float> $b */
    private static function pearsonr(array $a, array $b): float
    {
        $n = count($a);
        $ma = 0.0; $mb = 0.0;
        for ($i = 0; $i < $n; $i++) { $ma += $a[$i]; $mb += $b[$i]; }
        $ma /= $n; $mb /= $n;
        $num = 0.0; $da = 0.0; $db = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $x = $a[$i] - $ma; $y = $b[$i] - $mb;
            $num += $x * $y; $da += $x * $x; $db += $y * $y;
        }
        return $num / sqrt($da * $db);
    }

    /** Filliben uniform order-statistic medians. @return list<float> */
    private static function uom(int $n): array
    {
        $v = array_fill(0, $n, 0.0);
        $v[$n - 1] = 0.5 ** (1.0 / $n);
        $v[0] = 1.0 - $v[$n - 1];
        for ($i = 2; $i < $n; $i++) {
            $v[$i - 1] = ($i - 0.3175) / ($n + 0.365);
        }
        return $v;
    }

    /** Inverse standard-normal CDF (Acklam's rational approximation; ~1e-9, ample for the plot correlation). */
    private static function ndtri(float $p): float
    {
        static $a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
        static $b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
        static $c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
        static $d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
        $plow = 0.02425;
        if ($p < $plow) {
            $q = sqrt(-2.0 * log($p));
            return ((((($c[0] * $q + $c[1]) * $q + $c[2]) * $q + $c[3]) * $q + $c[4]) * $q + $c[5]) /
                   (((($d[0] * $q + $d[1]) * $q + $d[2]) * $q + $d[3]) * $q + 1.0);
        }
        if ($p <= 1.0 - $plow) {
            $q = $p - 0.5; $r = $q * $q;
            return ((((($a[0] * $r + $a[1]) * $r + $a[2]) * $r + $a[3]) * $r + $a[4]) * $r + $a[5]) * $q /
                   ((((($b[0] * $r + $b[1]) * $r + $b[2]) * $r + $b[3]) * $r + $b[4]) * $r + 1.0);
        }
        $q = sqrt(-2.0 * log(1.0 - $p));
        return -((((($c[0] * $q + $c[1]) * $q + $c[2]) * $q + $c[3]) * $q + $c[4]) * $q + $c[5]) /
                (((($d[0] * $q + $d[1]) * $q + $d[2]) * $q + $d[3]) * $q + 1.0);
    }
}
