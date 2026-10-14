<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Exceptions\TesseroException;

/**
 * scipy.stats.rv_histogram: a continuous distribution built from a histogram
 * (counts, bin_edges) -- a piecewise-constant pdf and a piecewise-linear cdf.
 *
 * Pure PHP, scalar-or-array in / same-shape out, deterministic (no RNG, no kernel call) --
 * mirrors the {@see \Tessero\LogRank} style: all arithmetic is reimplemented here so one class
 * serves both backends.
 *
 * Reproduces scipy's _continuous_distns.rv_histogram exactly:
 *   - with density=false (counts), the per-bin density is counts_i / width_i, then renormalised so
 *     sum(density_i * width_i) == 1; with density=true the counts are already a density and only the
 *     renormalisation is applied (for constant bin widths the two are equivalent).
 *   - _hpdf = [0, p_0, ..., p_{n-1}, 0]; pdf(x) = _hpdf[searchsorted(bins, x, side='right')].
 *   - _hcdf = [0, cumsum(p_i * width_i)]; cdf(x) = interp(x, bins, _hcdf) (clamped at the support).
 *   - ppf(q) = interp(q, _hcdf, bins); sf(x) = 1 - cdf(x).
 *   - mean / var / median follow from the step density in closed form (scipy's generic moment
 *     integration returns the same values to machine precision for a piecewise-constant pdf).
 *
 * The `density` argument matches scipy: null (default) assumes density=true when the bin widths vary
 * and density=false (counts) when they are constant; pass it explicitly to be unambiguous.
 */
final class RvHistogram
{
    /** @var list<float> normalised per-bin density p_i (length n) */
    private array $p;
    /** @var list<float> bin edges (length n+1) */
    private array $bins;
    /** @var list<float> [0, p_0, ..., p_{n-1}, 0] (length n+2) */
    private array $hpdf;
    /** @var list<float> [0, cumsum(p_i*width_i)] (length n+1) */
    private array $hcdf;
    /** @var list<float> bin widths (length n) */
    private array $widths;

    /**
     * @param list<float> $counts   per-bin content (np.histogram's first return)
     * @param list<float> $binEdges n+1 bin boundaries (np.histogram's second return)
     * @param bool|null   $density  null = scipy default (density when widths vary)
     */
    public function __construct(array $counts, array $binEdges, ?bool $density = null)
    {
        $c = array_map('floatval', array_values($counts));
        $b = array_map('floatval', array_values($binEdges));
        $n = count($c);
        if ($n < 1 || count($b) !== $n + 1) {
            throw new TesseroException('rv_histogram: expected n counts and n+1 bin edges.');
        }
        $widths = [];
        for ($i = 0; $i < $n; $i++) {
            $w = $b[$i + 1] - $b[$i];
            if (! ($w > 0.0) || ! is_finite($w)) {
                throw new TesseroException('rv_histogram: bin edges must be strictly increasing and finite.');
            }
            $widths[] = $w;
        }
        $varying = false;
        for ($i = 1; $i < $n; $i++) {
            // np.allclose default: atol 1e-8, rtol 1e-5
            if (abs($widths[$i] - $widths[0]) > 1e-8 + 1e-5 * abs($widths[0])) {
                $varying = true;
                break;
            }
        }
        if ($density === null) {
            $density = $varying;   // scipy: density=True when bins vary, else counts
        }

        $raw = $c;
        if (! $density) {
            for ($i = 0; $i < $n; $i++) {
                $raw[$i] = $c[$i] / $widths[$i];
            }
        }
        $norm = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $norm += $raw[$i] * $widths[$i];
        }
        if (! ($norm > 0.0)) {
            throw new TesseroException('rv_histogram: histogram has zero total mass.');
        }
        $p = [];
        for ($i = 0; $i < $n; $i++) {
            $p[$i] = $raw[$i] / $norm;
        }

        $hpdf = [0.0];
        foreach ($p as $pi) {
            $hpdf[] = $pi;
        }
        $hpdf[] = 0.0;

        $hcdf = [0.0];
        $acc = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $acc += $p[$i] * $widths[$i];
            $hcdf[] = $acc;
        }

        $this->p = $p;
        $this->bins = $b;
        $this->widths = $widths;
        $this->hpdf = $hpdf;
        $this->hcdf = $hcdf;
    }

    /** @param list<float>|tuple{list<float>, list<float>} $histogram */
    public static function fromHistogram(array $counts, array $binEdges, ?bool $density = null): self
    {
        return new self($counts, $binEdges, $density);
    }

    /** pdf(x) = _hpdf[searchsorted(bins, x, side='right')] -- scalar or array in, same out. */
    public function pdf(mixed $x): mixed
    {
        return $this->map($x, function (float $xi): float {
            $idx = self::searchsortedRight($this->bins, $xi);
            return $this->hpdf[$idx];
        });
    }

    /** cdf(x) = interp(x, bins, _hcdf). */
    public function cdf(mixed $x): mixed
    {
        return $this->map($x, fn (float $xi): float => self::interp($xi, $this->bins, $this->hcdf));
    }

    /** sf(x) = 1 - cdf(x). */
    public function sf(mixed $x): mixed
    {
        return $this->map($x, fn (float $xi): float => 1.0 - self::interp($xi, $this->bins, $this->hcdf));
    }

    /** ppf(q) = interp(q, _hcdf, bins). */
    public function ppf(mixed $q): mixed
    {
        return $this->map($q, fn (float $qi): float => self::interp($qi, $this->hcdf, $this->bins));
    }

    /** E[X] = sum_i p_i * (b_{i+1}^2 - b_i^2) / 2. */
    public function mean(): float
    {
        $m = 0.0;
        $n = count($this->p);
        for ($i = 0; $i < $n; $i++) {
            $a = $this->bins[$i];
            $b = $this->bins[$i + 1];
            $m += $this->p[$i] * ($b * $b - $a * $a) / 2.0;
        }
        return $m;
    }

    /** Var[X] = E[X^2] - E[X]^2, E[X^2] = sum_i p_i * (b_{i+1}^3 - b_i^3) / 3. */
    public function var(): float
    {
        $mean = $this->mean();
        $m2 = 0.0;
        $n = count($this->p);
        for ($i = 0; $i < $n; $i++) {
            $a = $this->bins[$i];
            $b = $this->bins[$i + 1];
            $m2 += $this->p[$i] * ($b * $b * $b - $a * $a * $a) / 3.0;
        }
        return $m2 - $mean * $mean;
    }

    public function std(): float
    {
        return sqrt($this->var());
    }

    public function median(): float
    {
        return self::interp(0.5, $this->hcdf, $this->bins);
    }

    /**
     * numpy.searchsorted(a, v, side='right'): number of entries a_j <= v.
     *
     * @param list<float> $a strictly increasing
     */
    private static function searchsortedRight(array $a, float $v): int
    {
        $lo = 0;
        $hi = count($a);
        while ($lo < $hi) {
            $mid = intdiv($lo + $hi, 2);
            if ($v < $a[$mid]) {
                $hi = $mid;
            } else {
                $lo = $mid + 1;
            }
        }
        return $lo;
    }

    /**
     * numpy.interp(x, xp, fp): piecewise-linear, clamped to fp[0] / fp[-1] outside [xp[0], xp[-1]].
     * For ties in xp (flat segments) numpy returns fp at the first matching xp; replicated via
     * searchsorted(side='right') then stepping back to the left segment, matching np.interp.
     *
     * @param list<float> $xp strictly non-decreasing
     * @param list<float> $fp same length as xp
     */
    private static function interp(float $x, array $xp, array $fp): float
    {
        $n = count($xp);
        if ($x <= $xp[0]) {
            return $fp[0];
        }
        if ($x >= $xp[$n - 1]) {
            return $fp[$n - 1];
        }
        // find j with xp[j] <= x < xp[j+1]; numpy uses the rightmost such left node
        $idx = self::searchsortedRight($xp, $x);  // first index with xp[idx] > x, in [1, n-1]
        $j = $idx - 1;
        $x0 = $xp[$j];
        $x1 = $xp[$j + 1];
        $f0 = $fp[$j];
        $f1 = $fp[$j + 1];
        $dx = $x1 - $x0;
        if ($dx == 0.0) {
            return $f0;
        }
        return $f0 + ($f1 - $f0) * (($x - $x0) / $dx);
    }

    /**
     * Apply $f elementwise; scalar in -> scalar out, list in -> list out (nested lists preserved).
     */
    private function map(mixed $x, callable $f): mixed
    {
        if (is_array($x)) {
            $out = [];
            foreach ($x as $k => $v) {
                $out[$k] = $this->map($v, $f);
            }
            return $out;
        }
        return $f((float) $x);
    }
}
