<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Random\Generator;
use Tessero\Special;

/**
 * scipy.stats.power, pure PHP over Tessero's PCG64 Generator and the landed
 * scipy.special Student-t CDF (stdtr). This estimates the power of a hypothesis
 * test by simulation and reproduces scipy bit-for-bit when:
 *
 *   - each sample set is drawn by one rvs_i(size=[n_resamples, n_obs_i]) call per
 *     sample, in order (single batch: batch defaults to n_resamples), using a
 *     Tessero Generator draw that matches numpy's exactly (standard_normal here);
 *   - the test is a deterministic hypothesis test applied along the last axis,
 *     returning a p-value per resample;
 *   - power = mean(pvalues < significance).
 *
 * scipy.stats.power does NOT pass the rng into the user rvs; the rvs closes over a
 * seeded default_rng. We mirror that: one Generator seeded with $seed drives every
 * rvs draw, consumed in the order the samples appear.
 *
 * Supported deterministic tests (closed-form p-value): 'ttest_1samp' (one sample
 * vs a population mean, two-sided) and 'ttest_ind' (two independent samples, equal
 * variance / pooled, two-sided) -- the common building blocks for power studies.
 *
 * Returns ['power' => float, 'pvalues' => list<float>].
 */
final class Power
{
    /**
     * @param string             $test       'ttest_1samp' | 'ttest_ind'
     * @param list<array{0:string,1:array<string,float>}> $rvs  one [kind, params] per sample;
     *                                        kind 'standard_normal' | 'normal'
     * @param list<int>          $nObservations one sample size per rvs
     * @param array<string,float> $testKwargs extra test args (e.g. ['popmean' => 0.0] for ttest_1samp)
     */
    public static function power(
        string $test,
        array $rvs,
        array $nObservations,
        float $significance = 0.01,
        int $nResamples = 10000,
        int $seed = 0,
        array $testKwargs = []
    ): array {
        if (count($rvs) !== count($nObservations)) {
            throw new \InvalidArgumentException('power: len(rvs) must equal len(n_observations).');
        }
        if ($test === 'ttest_1samp' && count($rvs) !== 1) {
            throw new \InvalidArgumentException('power: ttest_1samp needs exactly one sample.');
        }
        if ($test === 'ttest_ind' && count($rvs) !== 2) {
            throw new \InvalidArgumentException('power: ttest_ind needs exactly two samples.');
        }
        if (! in_array($test, ['ttest_1samp', 'ttest_ind'], true)) {
            throw new \InvalidArgumentException("power: unsupported test '{$test}'.");
        }

        // One Generator drives all rvs draws, consumed sample-by-sample (single batch).
        $gen = Generator::defaultRng($seed);
        $samples = [];
        foreach ($rvs as $i => [$kind, $params]) {
            $n = $nObservations[$i];
            $samples[] = self::drawRvs($gen, $kind, $params, $nResamples, $n);
        }

        $pvalues = match ($test) {
            'ttest_1samp' => self::ttest1Samp($samples[0], (float) ($testKwargs['popmean'] ?? 0.0)),
            'ttest_ind'   => self::ttestInd($samples[0], $samples[1]),
            default       => throw new \InvalidArgumentException("test '{$test}'"),
        };

        $rej = 0;
        foreach ($pvalues as $p) {
            if ($p < $significance) {
                $rej++;
            }
        }
        $power = $rej / count($pvalues);

        return ['power' => $power, 'pvalues' => $pvalues];
    }

    /**
     * Draw an (rows, n) block in numpy row-major order.
     *
     * @param array<string,float> $params
     * @return list<list<float>>
     */
    private static function drawRvs(Generator $gen, string $kind, array $params, int $rows, int $n): array
    {
        return match ($kind) {
            'standard_normal' => $gen->standardNormal([$rows, $n])->toArray(),
            'normal'          => $gen->normal($params['loc'] ?? 0.0, $params['scale'] ?? 1.0, [$rows, $n])->toArray(),
            default           => throw new \InvalidArgumentException("power: unknown rvs kind '{$kind}'."),
        };
    }

    /**
     * Two-sided one-sample t-test p-value per row.
     *
     * @param list<list<float>> $samp
     * @return list<float>
     */
    private static function ttest1Samp(array $samp, float $popmean): array
    {
        $out = [];
        foreach ($samp as $row) {
            $n = count($row);
            $mean = array_sum($row) / $n;
            $ss = 0.0;
            foreach ($row as $v) {
                $d = $v - $mean;
                $ss += $d * $d;
            }
            $sd = sqrt($ss / ($n - 1));                 // ddof=1
            $t = ($mean - $popmean) / ($sd / sqrt($n));
            $out[] = self::tTwoSided($t, $n - 1);
        }

        return $out;
    }

    /**
     * Two-sided two-sample (pooled / equal-variance) t-test p-value per row.
     *
     * @param list<list<float>> $a
     * @param list<list<float>> $b
     * @return list<float>
     */
    private static function ttestInd(array $a, array $b): array
    {
        $R = count($a);
        $out = [];
        for ($r = 0; $r < $R; $r++) {
            $n1 = count($a[$r]);
            $n2 = count($b[$r]);
            $ma = array_sum($a[$r]) / $n1;
            $mb = array_sum($b[$r]) / $n2;
            $va = self::varDdof1($a[$r], $ma);
            $vb = self::varDdof1($b[$r], $mb);
            $df = $n1 + $n2 - 2;
            $svar = (($n1 - 1) * $va + ($n2 - 1) * $vb) / $df;
            $denom = sqrt($svar * (1.0 / $n1 + 1.0 / $n2));
            $t = ($ma - $mb) / $denom;
            $out[] = self::tTwoSided($t, $df);
        }

        return $out;
    }

    /** @param list<float> $x */
    private static function varDdof1(array $x, float $mean): float
    {
        $s = 0.0;
        foreach ($x as $v) {
            $d = $v - $mean;
            $s += $d * $d;
        }

        return $s / (count($x) - 1);
    }

    /** Two-sided Student-t p-value: 2 * sf(|t|, df) = 2 * (1 - stdtr(df, |t|)). */
    private static function tTwoSided(float $t, int $df): float
    {
        $sf = 1.0 - (float) Special::stdtr((float) $df, abs($t));
        $p = 2.0 * $sf;

        return $p > 1.0 ? 1.0 : $p;
    }
}
