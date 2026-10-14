<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Random\Generator;
use Tessero\Special;

/**
 * scipy.stats.goodness_of_fit, pure PHP over Tessero's PCG64 Generator and the
 * landed scipy.special primitives (ndtr, log_ndtr, betaincinv). This is a Monte
 * Carlo goodness-of-fit test: it reproduces scipy bit-for-bit for the subset of
 * configurations that avoid numerical distribution fitting:
 *
 *   - dist = 'norm' : the only distribution scipy fits with a closed form
 *     (loc = mean, scale = std(ddof=1)); the unknown parameters are estimated
 *     analytically for the data and for every Monte Carlo sample, so there is no
 *     irreproducible optimizer in the loop.
 *   - any distribution with ALL parameters supplied in known_params ("all fixed"):
 *     no fit step at all; samples are drawn from the fully specified distribution.
 *
 * The Monte Carlo null distribution is drawn by exactly one rvs(size=[n_mc, n])
 * call on the null-hypothesized distribution, matching
 * scipy.stats.monte_carlo_test's single-batch draw. For 'norm' this is
 *   loc + scale * Generator::standardNormal([n_mc, n])
 * and for an all-fixed 'expon' it is
 *   loc + scale * Generator::standardExponential([n_mc, n]),
 * both of which Tessero's Generator reproduces bit-for-bit against numpy's
 * default_rng. The p-value is scipy's (b + 1) / (n_mc + 1) with the same
 * gamma = |100 * eps * observed| tie tolerance.
 *
 * Supported statistics (deterministic, closed-form): 'ks', 'cvm', 'ad'
 * (alternative 'greater') and 'filliben' (alternative 'less').
 *
 * Returns ['statistic' => float, 'pvalue' => float, 'null_distribution' => list<float>,
 *          'params' => list<float>] where params are the fitted (or fixed)
 * null-hypothesized distribution parameters [loc, scale].
 */
final class GoodnessOfFit
{
    private const EPS100 = 2.220446049250313e-16 * 100.0;

    /**
     * @param list<float>              $data        1-D sample
     * @param string                   $dist        'norm' or 'expon'
     * @param string                   $statistic   'ad' | 'ks' | 'cvm' | 'filliben'
     * @param array<string,float|null> $knownParams ['loc' => ?, 'scale' => ?]; a value fixes that parameter
     */
    public static function goodnessOfFit(
        array $data,
        string $dist = 'norm',
        string $statistic = 'ad',
        array $knownParams = [],
        int $nMcSamples = 9999,
        int $seed = 0
    ): array {
        $data = array_map('floatval', array_values($data));
        $n = count($data);
        if ($n < 2) {
            throw new \InvalidArgumentException('goodness_of_fit: need at least two observations.');
        }
        if (! in_array($statistic, ['ad', 'ks', 'cvm', 'filliben'], true)) {
            throw new \InvalidArgumentException("goodness_of_fit: unsupported statistic '{$statistic}'.");
        }
        if (! in_array($dist, ['norm', 'expon'], true)) {
            throw new \InvalidArgumentException("goodness_of_fit: unsupported dist '{$dist}' (only 'norm' with closed-form fit and all-fixed distributions are reproducible).");
        }

        $fixedLoc = array_key_exists('loc', $knownParams) ? (float) $knownParams['loc'] : null;
        $fixedScale = array_key_exists('scale', $knownParams) ? (float) $knownParams['scale'] : null;
        $allFixed = ($fixedLoc !== null) && ($fixedScale !== null);

        if ($dist === 'expon' && ! $allFixed) {
            // scipy would fit expon numerically (expon has no closed-form entry in
            // goodness_of_fit's _fit_funs), which is not reproducible here.
            throw new \InvalidArgumentException("goodness_of_fit: dist='expon' is only reproducible when loc and scale are both in known_params.");
        }

        // Fit (or fix) the null-hypothesized distribution parameters for the data.
        [$nhdLoc, $nhdScale] = self::fit($dist, $data, $fixedLoc, $fixedScale);

        // Observed statistic.
        $observed = self::statistic($dist, $statistic, [$data], $fixedLoc, $fixedScale)[0];

        // Monte Carlo null distribution: one rvs(size=[n_mc, n]) draw from the
        // null-hypothesized distribution, in numpy row-major order.
        $gen = Generator::defaultRng($seed);
        $std = $dist === 'norm'
            ? $gen->standardNormal([$nMcSamples, $n])->toArray()
            : $gen->standardExponential([$nMcSamples, $n])->toArray();
        $draws = [];
        foreach ($std as $row) {
            $r = [];
            foreach ($row as $z) {
                $r[] = $nhdLoc + $nhdScale * $z;
            }
            $draws[] = $r;
        }

        $null = self::statistic($dist, $statistic, $draws, $fixedLoc, $fixedScale);

        // p-value (Phipson & Smyth) with scipy's tie tolerance.
        $alternative = $statistic === 'filliben' ? 'less' : 'greater';
        $gamma = abs(self::EPS100 * $observed);
        $count = 0;
        foreach ($null as $v) {
            if ($alternative === 'greater') {
                if ($v >= $observed - $gamma) {
                    $count++;
                }
            } else {
                if ($v <= $observed + $gamma) {
                    $count++;
                }
            }
        }
        $p = ($count + 1.0) / ($nMcSamples + 1.0);
        $p = max(0.0, min(1.0, $p));

        return [
            'statistic' => $observed,
            'pvalue' => $p,
            'null_distribution' => $null,
            'params' => [$nhdLoc, $nhdScale],
        ];
    }

    /**
     * Closed-form parameter fit for a single 1-D sample (matches scipy's
     * _fit_norm; for all-fixed distributions it simply returns the fixed values).
     *
     * @param list<float> $d
     * @return array{0: float, 1: float}
     */
    private static function fit(string $dist, array $d, ?float $fixedLoc, ?float $fixedScale): array
    {
        if ($dist === 'expon') {
            // Only reached when all-fixed.
            return [$fixedLoc, $fixedScale];
        }

        // norm
        if ($fixedLoc === null && $fixedScale === null) {
            $loc = self::mean($d);
            $scale = self::stdDdof1($d, $loc);
        } elseif ($fixedLoc === null) {
            $loc = self::mean($d);
            $scale = $fixedScale;
        } elseif ($fixedScale === null) {
            $loc = $fixedLoc;
            // scale = sqrt(mean((d - loc)^2))  (population, ddof=0)
            $s = 0.0;
            foreach ($d as $v) {
                $t = $v - $loc;
                $s += $t * $t;
            }
            $scale = sqrt($s / count($d));
        } else {
            $loc = $fixedLoc;
            $scale = $fixedScale;
        }

        return [$loc, $scale];
    }

    /**
     * Compute the comparison statistic for every row of $samples (each a 1-D
     * array of length n). For 'norm' with free parameters the distribution is
     * re-fit to each row (as scipy does); for an all-fixed distribution the
     * fixed parameters are used throughout.
     *
     * @param list<list<float>> $samples
     * @return list<float>
     */
    private static function statistic(
        string $dist,
        string $stat,
        array $samples,
        ?float $fixedLoc,
        ?float $fixedScale
    ): array {
        $out = [];
        foreach ($samples as $sample) {
            $x = $sample;
            sort($x);                                   // ascending; numpy np.sort
            $m = count($x);

            [$loc, $scale] = self::fit($dist, $sample, $fixedLoc, $fixedScale);

            // standardized, sorted values z_i = (x_i - loc) / scale
            $z = [];
            foreach ($x as $xi) {
                $z[] = ($xi - $loc) / $scale;
            }

            $out[] = match ($stat) {
                'ks'       => self::ksStat($dist, $z, $m),
                'cvm'      => self::cvmStat($dist, $z, $m),
                'ad'       => self::adStat($dist, $z, $m),
                'filliben' => self::fillibenStat($dist, $x, $loc, $scale, $m),
                default    => throw new \InvalidArgumentException("stat '{$stat}'"),
            };
        }

        return $out;
    }

    /** cdf of the standardized distribution at a single z. */
    private static function cdf(string $dist, float $z): float
    {
        if ($dist === 'expon') {
            return $z <= 0.0 ? 0.0 : 1.0 - exp(-$z);
        }

        return (float) Special::ndtr($z);
    }

    /** log cdf of the standardized distribution at a single z (for 'ad'). */
    private static function logCdf(string $dist, float $z): float
    {
        if ($dist === 'expon') {
            return log($z <= 0.0 ? 0.0 : 1.0 - exp(-$z));
        }

        return (float) Special::logNdtr($z);
    }

    /** log sf of the standardized distribution at a single z (for 'ad'). */
    private static function logSf(string $dist, float $z): float
    {
        if ($dist === 'expon') {
            return $z <= 0.0 ? 0.0 : -$z;    // sf = exp(-z) => log sf = -z
        }

        return (float) Special::logNdtr(-$z);  // norm.logsf(z) == log_ndtr(-z)
    }

    /**
     * @param list<float> $z sorted standardized values
     */
    private static function ksStat(string $dist, array $z, int $m): float
    {
        // Dplus = max_i( (i+1)/m - cdf_i ); Dminus = max_i( cdf_i - i/m )
        $dplus = -INF;
        $dminus = -INF;
        for ($i = 0; $i < $m; $i++) {
            $c = self::cdf($dist, $z[$i]);
            $dp = ($i + 1) / $m - $c;
            $dm = $c - $i / $m;
            if ($dp > $dplus) {
                $dplus = $dp;
            }
            if ($dm > $dminus) {
                $dminus = $dm;
            }
        }

        return max($dplus, $dminus);
    }

    /**
     * @param list<float> $z sorted standardized values
     */
    private static function cvmStat(string $dist, array $z, int $m): float
    {
        $w = 1.0 / (12.0 * $m);
        for ($i = 0; $i < $m; $i++) {
            $u = (2.0 * ($i + 1) - 1.0) / (2.0 * $m);
            $c = self::cdf($dist, $z[$i]);
            $d = $u - $c;
            $w += $d * $d;
        }

        return $w;
    }

    /**
     * Anderson-Darling: -m - sum_i (2i-1)/m * (logcdf(z_i) + logsf(z_{m-1-i})).
     *
     * @param list<float> $z sorted standardized values
     */
    private static function adStat(string $dist, array $z, int $m): float
    {
        $s = 0.0;
        for ($i = 0; $i < $m; $i++) {
            $w = (2.0 * ($i + 1) - 1.0) / $m;
            $s += $w * (self::logCdf($dist, $z[$i]) + self::logSf($dist, $z[$m - 1 - $i]));
        }

        return -$m - $s;
    }

    /**
     * Filliben: correlation between sorted data X and the theoretical order
     * statistic medians M = dist.ppf(beta(k, m+1-k).median()) (with loc/scale).
     * Only 'norm' is supported (ppf = loc + scale * ndtri).
     *
     * @param list<float> $x sorted raw sample
     */
    private static function fillibenStat(string $dist, array $x, float $loc, float $scale, int $m): float
    {
        if ($dist !== 'norm') {
            throw new \InvalidArgumentException("goodness_of_fit: filliben statistic supported only for dist='norm'.");
        }
        $M = [];
        for ($k = 1; $k <= $m; $k++) {
            $med = (float) Special::betaincinv((float) $k, (float) ($m + 1 - $k), 0.5);
            $M[] = $loc + $scale * (float) Special::ndtri($med);
        }

        $xm = self::mean($x);
        $mm = self::mean($M);
        $num = 0.0;
        $sx = 0.0;
        $sM = 0.0;
        for ($i = 0; $i < $m; $i++) {
            $a = $x[$i] - $xm;
            $b = $M[$i] - $mm;
            $num += $a * $b;
            $sx += $a * $a;
            $sM += $b * $b;
        }

        return $num / sqrt($sx * $sM);
    }

    /** @param list<float> $x */
    private static function mean(array $x): float
    {
        return array_sum($x) / count($x);
    }

    /** sample std with ddof=1 about a precomputed mean. @param list<float> $x */
    private static function stdDdof1(array $x, float $mean): float
    {
        $s = 0.0;
        foreach ($x as $v) {
            $d = $v - $mean;
            $s += $d * $d;
        }

        return sqrt($s / (count($x) - 1));
    }
}
