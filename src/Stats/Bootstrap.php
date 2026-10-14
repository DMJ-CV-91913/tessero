<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Random\Generator;
use Tessero\Special;

/**
 * scipy.stats.bootstrap, pure PHP over a PHP callable statistic and Tessero's
 * PCG64 Generator. Bit-exact against scipy when the same default_rng seed is used:
 *
 *   - resample indices come from Generator::integers(0, n, [n_resamples, n]), which
 *     reproduces numpy rng.integers draws exactly, in scipy's draw order (one call
 *     per sample, in sequence; single batch since batch defaults to n_resamples);
 *   - the confidence interval uses numpy's 'linear' (type-7) quantile and the same
 *     percentile / basic / BCa formulas as scipy.stats._resampling;
 *   - standard_error is the ddof=1 std of the bootstrap distribution.
 *
 * The statistic is a PHP callable accepting len(data) resample arrays (each a flat
 * list<float>) and returning a float. Returns ['confidence_interval' => [low, high],
 * 'bootstrap_distribution' => list<float>, 'standard_error' => float].
 */
final class Bootstrap
{
    public static function bootstrap(
        array $data,
        callable $statistic,
        int $nResamples = 9999,
        string $method = 'percentile',
        float $confidenceLevel = 0.95,
        string $alternative = 'two-sided',
        int|array|Generator|null $rng = null
    ): array {
        $method = strtolower($method);
        if (! in_array($method, ['percentile', 'basic', 'bca'], true)) {
            throw new \InvalidArgumentException("bootstrap: unknown method '{$method}'.");
        }
        if (! in_array($alternative, ['two-sided', 'less', 'greater'], true)) {
            throw new \InvalidArgumentException("bootstrap: unknown alternative '{$alternative}'.");
        }
        $data = array_map(static fn ($x) => array_map('floatval', array_values($x)), array_values($data));
        $gen = ($rng instanceof Generator) ? $rng : Generator::defaultRng($rng);

        // Bootstrap distribution: single batch, one integers() draw per sample in order.
        // integers(0, n, [nResamples, n])->toArray() is a list of nResamples index rows.
        $thetaHatB = [];
        $idxPerSample = [];
        foreach ($data as $sample) {
            $idxPerSample[] = $gen->integers(0, count($sample), [$nResamples, count($sample)])->toArray();
        }
        for ($r = 0; $r < $nResamples; $r++) {
            $resamples = [];
            foreach ($data as $s => $sample) {
                $row = [];
                foreach ($idxPerSample[$s][$r] as $idx) {
                    $row[] = $sample[$idx];
                }
                $resamples[] = $row;
            }
            $thetaHatB[] = (float) $statistic(...$resamples);
        }

        $alpha = ($alternative === 'two-sided')
            ? (1.0 - $confidenceLevel) / 2.0
            : (1.0 - $confidenceLevel);

        if ($method === 'bca') {
            [$a1, $a2] = self::bcaInterval($data, $statistic, $alpha, $thetaHatB);
            $ciL = self::quantileLinear($thetaHatB, $a1);
            $ciU = self::quantileLinear($thetaHatB, $a2);
        } else {
            $ciL = self::quantileLinear($thetaHatB, $alpha);
            $ciU = self::quantileLinear($thetaHatB, 1.0 - $alpha);
            if ($method === 'basic') {
                $thetaHat = (float) $statistic(...$data);
                [$ciL, $ciU] = [2.0 * $thetaHat - $ciU, 2.0 * $thetaHat - $ciL];
            }
        }

        if ($alternative === 'less') {
            $ciL = -INF;
        } elseif ($alternative === 'greater') {
            $ciU = INF;
        }

        return [
            'confidence_interval' => [$ciL, $ciU],
            'bootstrap_distribution' => $thetaHatB,
            'standard_error' => self::stdDdof1($thetaHatB),
        ];
    }

    /** numpy.quantile with method='linear' (Hyndman & Fan type 7), the scipy.stats.quantile default. */
    private static function quantileLinear(array $a, float $q): float
    {
        $s = $a;
        sort($s);
        $n = count($s);
        if ($n === 1) {
            return (float) $s[0];
        }
        $pos = $q * ($n - 1);
        $lo = (int) floor($pos);
        $hi = (int) ceil($pos);
        if ($lo === $hi) {
            return (float) $s[$lo];
        }
        $frac = $pos - $lo;
        return $s[$lo] + ($s[$hi] - $s[$lo]) * $frac;
    }

    private static function stdDdof1(array $a): float
    {
        $n = count($a);
        if ($n < 2) {
            return NAN;
        }
        $mean = array_sum($a) / $n;
        $ss = 0.0;
        foreach ($a as $v) {
            $d = $v - $mean;
            $ss += $d * $d;
        }
        return sqrt($ss / ($n - 1));
    }

    /**
     * Bias-corrected and accelerated interval probabilities (alpha_1, alpha_2),
     * following scipy.stats._resampling._bca_interval for one or more samples.
     */
    private static function bcaInterval(array $data, callable $statistic, float $alpha, array $thetaHatB): array
    {
        $thetaHat = (float) $statistic(...$data);
        $B = count($thetaHatB);
        // _percentile_of_score with the 'mean' convention
        $nz = 0;
        foreach ($thetaHatB as $v) {
            if ($v < $thetaHat) {
                $nz++;
            }
            if ($v <= $thetaHat) {
                $nz++;
            }
        }
        $percentile = $nz / (2 * $B);
        $z0 = (float) Special::ndtri($percentile);

        // acceleration a_hat from the jackknife over every sample
        $sumNum = 0.0;
        $sumDen = 0.0;
        foreach ($data as $j => $sample) {
            $nj = count($sample);
            $thetaHatI = [];
            for ($leave = 0; $leave < $nj; $leave++) {
                $args = $data;
                $jk = [];
                for ($i = 0; $i < $nj; $i++) {
                    if ($i !== $leave) {
                        $jk[] = $sample[$i];
                    }
                }
                $args[$j] = $jk;
                $thetaHatI[] = (float) $statistic(...$args);
            }
            $dot = array_sum($thetaHatI) / $nj;
            $s3 = 0.0;
            $s2 = 0.0;
            foreach ($thetaHatI as $t) {
                $u = ($nj - 1) * ($dot - $t);
                $s3 += $u * $u * $u;
                $s2 += $u * $u;
            }
            $sumNum += $s3 / ($nj ** 3);
            $sumDen += $s2 / ($nj ** 2);
        }
        $aHat = (1.0 / 6.0) * $sumNum / ($sumDen ** 1.5);

        $zAlpha = (float) Special::ndtri($alpha);
        $z1Alpha = -$zAlpha;
        $num1 = $z0 + $zAlpha;
        $a1 = (float) Special::ndtr($z0 + $num1 / (1.0 - $aHat * $num1));
        $num2 = $z0 + $z1Alpha;
        $a2 = (float) Special::ndtr($z0 + $num2 / (1.0 - $aHat * $num2));

        return [$a1, $a2];
    }
}
